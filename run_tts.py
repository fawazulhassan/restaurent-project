"""Phase 4 TTS test CLI — synthesize Roman Urdu text to speech."""

import argparse
import sys
import time

import config
from app.order import Order, add_item_to_order, build_confirmation_roman, confirm_order
from app.tts import get_tts_model, play_audio, synthesize, synthesize_to_file

DEFAULT_GREETING = "Assalam o alaikum, aap ka kya order hai?"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def sample_confirmation_order() -> Order:
    order = Order()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    order.delivery_address = "Model Town, Kasur, house 45"
    order.special_instructions = "Extra spicy, no salad"
    confirm_order(order)
    return order


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(description="Phase 4 TTS test CLI")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Roman Urdu text to synthesize",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Save WAV to this path (no playback)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Synthesize a sample order confirmation",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Print synthesis time",
    )
    args = parser.parse_args()

    if args.confirm:
        text = build_confirmation_roman(sample_confirmation_order())
    elif args.text:
        text = args.text
    else:
        text = DEFAULT_GREETING

    print("Loading TTS model...")
    get_tts_model()

    print(f"Synthesizing: {text}")
    start = time.perf_counter()
    if args.file:
        out_path = synthesize_to_file(text, args.file)
        print(f"Saved: {out_path}")
    else:
        out_path = synthesize_to_file(text)
        wav_bytes = out_path.read_bytes()
        print(f"Saved: {out_path}")
    elapsed = time.perf_counter() - start

    if args.latency or elapsed > 8:
        print(f"Latency: {elapsed:.1f}s")

    if not args.file:
        play_audio(wav_bytes)


if __name__ == "__main__":
    main()
