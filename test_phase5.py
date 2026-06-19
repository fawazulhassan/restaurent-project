"""Phase 5 agent smoke tests.

Manual mic checklist (run: python run_mic.py):
  5.1 Full voice order (2 karahi + address + confirm) -> order JSON saved, confirmation played
  5.2 Order without address -> AI voice asks for address
  5.3 Invalid menu item -> AI explains and suggests alternatives
  5.4 Multi-turn (4-6 turns) -> coherent conversation, correct order state
  5.5 Roman Urdu + English speech -> STT + dialog understand
  5.6 AI reply quality -> TTS speaks understandable Roman Urdu
  5.7 One turn latency -> STT + LLM + TTS under ~15s on CPU (use --latency)
  5.8 After confirm -> valid JSON in data/orders/
"""

import sys
import time

reconfigure = getattr(sys.stdout, "reconfigure", None)
if reconfigure is not None:
    reconfigure(encoding="utf-8")

import config
from app.agent import preload_models, process_text_turn, reply_text_for_tts
from app.dialog import build_system_prompt
from app.order import Order, OrderStatus, add_item_to_order, confirm_order

PASS = 0
FAIL = 0
SKIP = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} — {detail}")


def skip(name: str, reason: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  SKIP  {name} — {reason}")


def _api_key_ready() -> bool:
    key = config.OPENROUTER_API_KEY
    return bool(key) and key != "your_openrouter_api_key_here"


def _is_riff_wav(wav_bytes: bytes) -> bool:
    return len(wav_bytes) >= 4 and wav_bytes[:4] == b"RIFF"


def test_5xa_reply_text_for_tts() -> None:
    print("\n=== 5.x-a reply_text_for_tts ===")
    order = Order()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    order.delivery_address = "Model Town, Kasur"
    confirm_order(order)

    tts_text = reply_text_for_tts("Your order is confirmed.", order, is_complete=True)
    check("5.x-a roman confirmation", tts_text.startswith("Aap ka order confirm"))
    check("5.x-a mentions total", "Rs" in tts_text)

    passthrough = reply_text_for_tts("Delivery address bata dein?", order, is_complete=False)
    check("5.x-a passthrough reply", passthrough == "Delivery address bata dein?")


def test_5xb_process_text_turn_live() -> None:
    print("\n=== 5.x-b process_text_turn (live API) ===")
    if not _api_key_ready():
        skip("5.x-b", "OPENROUTER_API_KEY not set")
        return

    preload_models()
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]

    reply, audio, order, messages, is_complete = process_text_turn(
        "2 chicken karahi full", order, messages
    )
    check("5.x-b reply not empty", bool(reply))
    check("5.x-b wav bytes", len(audio) > 0)
    check("5.x-b riff header", _is_riff_wav(audio))
    check("5.x-b items added", len(order.items) >= 1)
    check("5.x-b not complete yet", not is_complete)


def test_5xc_multi_turn_live() -> None:
    print("\n=== 5.x-c multi-turn confirm (live API) ===")
    if not _api_key_ready():
        skip("5.x-c", "OPENROUTER_API_KEY not set")
        return

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    turns = [
        "2 chicken karahi full",
        "Model Town Kasur house 45",
        "confirm",
    ]
    is_complete = False
    for turn in turns:
        reply, audio, order, messages, is_complete = process_text_turn(
            turn, order, messages
        )
        if is_complete:
            break

    check("5.x-c confirmed", is_complete and order.status == OrderStatus.CONFIRMED)
    check("5.x-c has address", bool(order.delivery_address))
    check("5.x-c wav on last turn", _is_riff_wav(audio))


def test_5xd_invalid_item_live() -> None:
    print("\n=== 5.x-d invalid item (live API) ===")
    if not _api_key_ready():
        skip("5.x-d", "OPENROUTER_API_KEY not set")
        return

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    reply, audio, order, messages, is_complete = process_text_turn(
        "3 zinger burger", order, messages
    )
    check("5.x-d no crash", True)
    check("5.x-d reply not empty", bool(reply))
    check("5.x-d no items added", len(order.items) == 0)
    check("5.x-d audio generated", _is_riff_wav(audio))
    check("5.x-d not complete", not is_complete)


def test_5xe_latency_log() -> None:
    print("\n=== 5.x-e latency log path ===")
    if not _api_key_ready():
        skip("5.x-e", "OPENROUTER_API_KEY not set")
        return

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    start = time.perf_counter()
    process_text_turn("ek naan", order, messages, log_latency=True)
    elapsed = time.perf_counter() - start
    check("5.x-e completed", elapsed > 0)
    if elapsed > 60:
        print(f"  NOTE  5.x-e slow CPU: {elapsed:.1f}s (informational only)")


def main() -> None:
    print("Phase 5 automated tests")
    test_5xa_reply_text_for_tts()
    test_5xb_process_text_turn_live()
    test_5xc_multi_turn_live()
    test_5xd_invalid_item_live()
    test_5xe_latency_log()

    print(f"\nResults: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
