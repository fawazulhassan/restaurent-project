"""Voice ordering demo: fixed-duration recording + full agent (STT + dialog + TTS)."""

import argparse

import config
from app.agent import play_greeting, preload_models, process_text_turn
# from app.cli_utils import configure_stdout, print_order_summary
from app.dialog import build_system_prompt
from app.dialog_state import DialogState
from app.llm import LLMRateLimitError, LLMServiceError
from app.order import Order, build_confirmation_english, save_order
from app.stt import record_and_transcribe
from app.tts import play_audio

QUIT_WORDS = frozenset({"quit", "exit", "band", "bye"})


def listen(seconds: float) -> str:
    print(f"Recording {seconds:.0f} seconds... speak now.")
    try:
        return record_and_transcribe(
            seconds,
            quiet=True,
            roman_bias=True,
            latinize=False,
        )
    except RuntimeError as e:
        print(f"Mic error: {e}")
        return ""


def main() -> None:
    # configure_stdout()

    parser = argparse.ArgumentParser(
        description="Voice order chat — fixed-duration recording + TTS replies"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=config.DEFAULT_RECORD_SECONDS,
        help="Seconds to record per turn (default: from config)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Log per-turn LLM/TTS timings",
    )
    parser.add_argument(
        "--no-greeting",
        action="store_true",
        help="Skip spoken opening greeting",
    )
    args = parser.parse_args()

    print("Assalam o alaikum! Welcome to Kasur Kitchen.")
    print("Speak in Roman Urdu or English. Press Enter to record each turn.")
    print("AI replies with voice. Ctrl+C to exit.\n")

    try:
        preload_models()
    except LLMServiceError as e:
        print(f"LLM startup error: {e}")
        return

    if not args.no_greeting:
        play_greeting()

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    dialog_state = DialogState()
    log_latency = args.latency or config.AGENT_LOG_LATENCY

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
            reply, reply_audio, order, messages, is_complete = process_text_turn(
                user_input,
                order,
                messages,
                dialog_state=dialog_state,
                log_latency=log_latency,
            )
        except LLMRateLimitError as e:
            print(f"\nRate limit: {e}\n")
            continue
        except LLMServiceError as e:
            print(f"\nLLM error: {e}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAI: {reply}\n")
        # print_order_summary(order)

        if reply_audio:
            play_audio(reply_audio)

        if is_complete:
            path = save_order(order)
            print(f"Order saved: {path}")
            print(f"\n{build_confirmation_english(order)}\n")
            break


if __name__ == "__main__":
    main()
