"""Phase 5 agent smoke tests.

Manual mic checklist (run: python run_mic.py):
  5.1 Full voice order (2 karahi + address + confirm) -> order JSON saved, confirmation played
  5.2 Order without address -> AI voice asks for address
  5.3 Invalid menu item -> AI explains and suggests alternatives
  5.4 Multi-turn (4-6 turns) -> coherent conversation, correct order state
  5.5 Roman Urdu + English speech -> STT + dialog understand
  5.6 AI reply quality -> Piper speaks understandable Urdu script
  5.7 One turn latency -> STT + LLM + TTS under ~15s on CPU (use --latency)
  5.8 After confirm -> valid JSON in data/orders/
"""

import sys
import time
from unittest.mock import patch

reconfigure = getattr(sys.stdout, "reconfigure", None)
if reconfigure is not None:
    reconfigure(encoding="utf-8")

import config
from app.agent import preload_models, process_text_turn, reply_text_for_tts
from app.dialog import build_system_prompt
from app.llm import LLMServiceError, check_llm_service
from app.order import Order, OrderStatus, add_item_to_order, confirm_order
from app.tts import TTSServiceError, check_tts_service

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


def _llm_ready() -> bool:
    try:
        check_llm_service()
        return True
    except LLMServiceError:
        return False


def _is_riff_wav(wav_bytes: bytes) -> bool:
    return len(wav_bytes) >= 4 and wav_bytes[:4] == b"RIFF"


def _piper_ready() -> bool:
    try:
        check_tts_service()
        return True
    except TTSServiceError:
        return False


def test_5xa_reply_text_for_tts() -> None:
    print("\n=== 5.x-a reply_text_for_tts ===")
    order = Order()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    order.delivery_address = "Model Town, Kasur"
    confirm_order(order)

    tts_text = reply_text_for_tts("Your order is confirmed.", order, is_complete=True)
    check("5.x-a Urdu confirmation", tts_text.startswith("آپ کا آرڈر کنفرم"))
    check("5.x-a mentions total", "کل" in tts_text)

    passthrough = reply_text_for_tts("ڈیلیوری کا پتہ بتائیں؟", order, is_complete=False)
    check("5.x-a passthrough reply", passthrough == "ڈیلیوری کا پتہ بتائیں؟")


def test_5xf_tts_failure_preserves_order_state() -> None:
    print("\n=== 5.x-f TTS failure preserves order ===")
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]

    def fake_chat_turn(user_message, current_order, current_messages):
        del user_message
        add_item_to_order(current_order, "chicken-karahi", "full", 1)
        return (
            "آپ کا آرڈر شامل ہو گیا ہے۔",
            current_order,
            current_messages,
            False,
        )

    with patch("app.agent.chat_turn", side_effect=fake_chat_turn), patch(
        "app.agent.synthesize",
        side_effect=TTSServiceError("Piper unavailable"),
    ):
        reply, audio, updated_order, _, complete = process_text_turn(
            "ek chicken karahi",
            order,
            messages,
        )

    check("5.x-f reply preserved", reply == "آپ کا آرڈر شامل ہو گیا ہے۔")
    check("5.x-f empty audio fallback", audio == b"")
    check("5.x-f order preserved", len(updated_order.items) == 1)
    check("5.x-f turn remains active", not complete)


def test_5xg_system_prompt_requests_urdu_script() -> None:
    print("\n=== 5.x-g Urdu-script system prompt ===")
    prompt = build_system_prompt()
    check("5.x-g requests Urdu script", "Urdu script" in prompt)
    check("5.x-g no Latin-only rule", "ONLY the Latin alphabet" not in prompt)


def test_5xb_process_text_turn_live() -> None:
    print("\n=== 5.x-b process_text_turn (live API) ===")
    if not _llm_ready():
        skip("5.x-b", f"{config.LLM_PROVIDER} LLM is not ready")
        return
    if not _piper_ready():
        skip("5.x-b", "Piper TTS server is not running")
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
    if not _llm_ready():
        skip("5.x-c", f"{config.LLM_PROVIDER} LLM is not ready")
        return
    if not _piper_ready():
        skip("5.x-c", "Piper TTS server is not running")
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
    if not _llm_ready():
        skip("5.x-d", f"{config.LLM_PROVIDER} LLM is not ready")
        return
    if not _piper_ready():
        skip("5.x-d", "Piper TTS server is not running")
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
    if not _llm_ready():
        skip("5.x-e", f"{config.LLM_PROVIDER} LLM is not ready")
        return
    if not _piper_ready():
        skip("5.x-e", "Piper TTS server is not running")
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
    test_5xf_tts_failure_preserves_order_state()
    test_5xg_system_prompt_requests_urdu_script()
    test_5xb_process_text_turn_live()
    test_5xc_multi_turn_live()
    test_5xd_invalid_item_live()
    test_5xe_latency_log()

    print(f"\nResults: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
