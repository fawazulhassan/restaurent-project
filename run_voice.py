"""Voice ordering demo: STT (Phase 3) + dialog (Phase 2). Replies in Roman Urdu like run_text.py."""

import argparse
import sys

import config
from app.dialog import LLMRateLimitError, build_system_prompt, chat_turn
from app.order import Order, build_confirmation_english, save_order
from app.stt import get_model, record_and_transcribe
from run_text import configure_stdout, print_order_summary

QUIT_WORDS = frozenset({"quit", "exit", "band", "bye"})


def listen(seconds: float) -> str:
    print(f"Recording {seconds:.0f} seconds... speak now.")
    try:
        return record_and_transcribe(seconds, quiet=True, roman_bias=True)
    except RuntimeError as e:
        print(f"Mic error: {e}")
        return ""


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Voice order chat — same brain as run_text.py"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=config.DEFAULT_RECORD_SECONDS,
        help="Seconds to record per turn (default: from config)",
    )
    args = parser.parse_args()

    print("Assalam o alaikum! Welcome to Kasur Kitchen.")
    print("Speak in Roman Urdu or English. Press Enter to record each turn.")
    print("AI replies in Roman Urdu (Latin letters). Ctrl+C to exit.\n")

    print("Loading Whisper model...")
    get_model()

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]

    while True:
        try:
            input("Press Enter to speak...")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = listen(args.seconds).strip()
        if not user_input:
            print("No speech detected. Try again.\n")
            continue

        print(f"You: {user_input}")

        if user_input.lower() in QUIT_WORDS:
            print("Bye!")
            break

        try:
            reply, order, messages, is_complete = chat_turn(
                user_input, order, messages
            )
        except LLMRateLimitError as e:
            print(f"\nRate limit: {e}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAI: {reply}\n")
        print_order_summary(order)

        if is_complete:
            path = save_order(order)
            print(f"Order saved: {path}")
            print(f"\n{build_confirmation_english(order)}\n")
            break


if __name__ == "__main__":
    main()
