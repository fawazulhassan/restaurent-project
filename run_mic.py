"""Phase 5 full mic demo — speak, hear AI reply, save order."""

import argparse

from openai import RateLimitError

import config
from app.agent import play_greeting, preload_models, process_text_turn
from app.cli_utils import configure_stdout, print_order_summary
from app.dialog import LLMRateLimitError, build_system_prompt
from app.order import Order, build_confirmation_english, save_order
from app.stt import record_push_to_talk
from app.tts import play_audio

QUIT_WORDS = frozenset({"quit", "exit", "band", "bye"})


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Full voice order demo — STT + dialog + TTS"
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Log per-turn STT/LLM/TTS timings",
    )
    parser.add_argument(
        "--no-greeting",
        action="store_true",
        help="Skip spoken opening greeting",
    )
    args = parser.parse_args()

    print("Assalam o alaikum! Welcome to Kasur Kitchen.")
    print("Press Enter to speak, then Enter again to stop. Ctrl+C to exit.\n")

    preload_models()

    if not args.no_greeting:
        play_greeting()

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    log_latency = args.latency or config.AGENT_LOG_LATENCY

    while True:
        try:
            input("Press Enter to speak...")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            user_text = record_push_to_talk().strip()
        except RuntimeError as e:
            print(f"Mic error: {e}")
            continue

        if not user_text:
            print("No speech detected. Try again.\n")
            continue

        print(f"You: {user_text}")

        if user_text.lower() in QUIT_WORDS:
            print("Bye!")
            break

        try:
            reply_text, reply_audio, order, messages, is_complete = process_text_turn(
                user_text, order, messages, log_latency=log_latency
            )
        except (LLMRateLimitError, RateLimitError) as e:
            print(f"\nRate limit: {e}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAI: {reply_text}\n")
        print_order_summary(order)

        if reply_audio:
            play_audio(reply_audio)

        if is_complete:
            path = save_order(order)
            print(f"Order saved: {path}")
            print(f"\n{build_confirmation_english(order)}\n")
            break


if __name__ == "__main__":
    main()
