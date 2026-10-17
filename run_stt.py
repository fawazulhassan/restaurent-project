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
from typing import Any

from app.stt import (
    SttInitError,
    preload_stt,
    record_and_transcribe,
    record_clip,
    transcribe_file,
    transcribe_file_timed,
)

import config


def configure_stdout() -> None:
    stdout = sys.stdout
    if hasattr(stdout, "reconfigure"):
        casted = stdout  # keep runtime check; cast for static type checker
        casted_any: Any = casted
        casted_any.reconfigure(encoding="utf-8")


def print_result(text: str) -> None:
    if text:
        print(f"Heard: {text}")
    else:
        print("No speech detected.")


def run_once(seconds: float, *, latency: bool) -> None:
    print(f"Recording {seconds:.0f} seconds... speak now.")
    start = time.perf_counter()
    text = record_and_transcribe(seconds, roman_bias=True, latinize=True)
    elapsed = time.perf_counter() - start
    print_result(text)
    if latency:
        print(
            f"[latency] recording_seconds={seconds:.3f} "
            f"transcription_seconds={max(elapsed - seconds, 0):.3f} "
            f"model={config.WHISPER_MODEL} device={config.WHISPER_DEVICE} "
            f"compute_type={config.WHISPER_COMPUTE_TYPE}"
        )
    print(f"Done in {elapsed:.1f}s")


def run_file(path: str, *, latency: bool) -> None:
    print(f"Transcribing file: {path}")
    preload_stt()
    text, elapsed = transcribe_file_timed(path, roman_bias=True, latinize=True)
    print_result(text)
    if latency:
        print(
            f"[latency] recording_seconds=0.000 "
            f"transcription_seconds={elapsed:.3f} "
            f"model={config.WHISPER_MODEL} device={config.WHISPER_DEVICE} "
            f"compute_type={config.WHISPER_COMPUTE_TYPE}"
        )
    print(f"Done in {elapsed:.1f}s")


def run_loop(seconds: float) -> None:
    preload_stt()
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

    parser = argparse.ArgumentParser(
        description=f"STT test CLI ({config.STT_PROVIDER}, no AI replies)"
    )
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
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Log STT latency and runtime configuration",
    )
    parser.add_argument(
        "--record-clip",
        type=str,
        default=None,
        metavar="NAME",
        help="Record data/test_audio/NAME.wav and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --record-clip to overwrite an existing WAV",
    )
    args = parser.parse_args()

    try:
        if args.record_clip:
            out = record_clip(
                f"data/test_audio/{args.record_clip}.wav",
                args.seconds,
                force=args.force,
            )
            print(f"Recorded clip: {out}")
            return

        if args.file:
            run_file(args.file, latency=args.latency)
            return

        if args.loop:
            run_loop(args.seconds)
            return

        preload_stt()
        print("STT only — no AI reply. Heard text is Roman Urdu (Latin letters).")
        print("For full voice chat use: python run_voice.py\n")
        run_once(args.seconds, latency=args.latency)
    except (SttInitError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
