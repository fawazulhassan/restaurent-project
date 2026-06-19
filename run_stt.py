# To create a test WAV for case 3.6:
#   python run_stt.py --seconds 5
#   (say "mujhe do chicken karahi full chahiye")
#   Save the recording manually, or re-run with --file after saving to:
#   data/test_audio/order-sample.wav
#
# Full voice ordering (STT + AI replies like run_text.py):
#   python run_voice.py

import argparse
import sys
import time

from app.stt import get_model, record_and_transcribe, transcribe_file

import config


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_result(text: str) -> None:
    if text:
        print(f"Heard: {text}")
    else:
        print("No speech detected.")


def run_once(seconds: float) -> None:
    print(f"Recording {seconds:.0f} seconds... speak now.")
    start = time.perf_counter()
    text = record_and_transcribe(seconds, roman_bias=True, latinize=True)
    elapsed = time.perf_counter() - start
    print_result(text)
    print(f"Done in {elapsed:.1f}s")


def run_file(path: str) -> None:
    print(f"Transcribing file: {path}")
    print("Loading Whisper model...")
    get_model()
    start = time.perf_counter()
    text = transcribe_file(path, roman_bias=True, latinize=True)
    elapsed = time.perf_counter() - start
    print_result(text)
    print(f"Done in {elapsed:.1f}s")


def run_loop(seconds: float) -> None:
    print("Loading Whisper model...")
    get_model()
    print("STT loop — Ctrl+C to exit.\n")
    while True:
        try:
            input("Press Enter to record...")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        print(f"Recording {seconds:.0f} seconds... speak now.")
        start = time.perf_counter()
        try:
            text = record_and_transcribe(seconds, roman_bias=True, latinize=True)
        except RuntimeError as e:
            print(f"Error: {e}")
            continue
        elapsed = time.perf_counter() - start
        print_result(text)
        print(f"Done in {elapsed:.1f}s\n")


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(description="Phase 3 STT test CLI (no AI replies)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=config.DEFAULT_RECORD_SECONDS,
        help="Recording duration in seconds (default: from config)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Transcribe a WAV/audio file instead of using the mic",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat: press Enter to record each turn",
    )
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
        return

    if args.loop:
        run_loop(args.seconds)
        return

    print("Loading Whisper model...")
    get_model()
    print("STT only — no AI reply. Heard text is Roman Urdu (Latin letters).")
    print("For full voice chat use: python run_voice.py\n")
    run_once(args.seconds)


if __name__ == "__main__":
    main()
