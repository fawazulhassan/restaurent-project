"""Phase 3 STT smoke test — skips if local test WAV is missing."""

from pathlib import Path

from app.stt import transcribe_file

TEST_WAV = Path(__file__).parent / "data" / "test_audio" / "order-sample.wav"
FOOD_KEYWORDS = ("chicken", "karahi", "pizza", "cheese", "biryani", "naan", "do", "mujhe")


def main() -> None:
    if not TEST_WAV.is_file():
        print(f"SKIP: no test WAV at {TEST_WAV}")
        print("Record one locally — see run_stt.py header comment.")
        return

    text = transcribe_file(TEST_WAV).lower()
    print(f"Transcribed: {text}")

    if not text:
        raise SystemExit("FAIL: empty transcription")

    if not any(kw in text for kw in FOOD_KEYWORDS):
        raise SystemExit(f"FAIL: expected a food-order keyword in: {text!r}")

    print("PASS: transcribe_file returned expected keywords")


if __name__ == "__main__":
    main()
