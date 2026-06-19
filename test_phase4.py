"""Phase 4 TTS smoke tests."""

import io
import time

import scipy.io.wavfile

from app.order import Order, add_item_to_order, build_confirmation_roman, confirm_order
from app.tts import DEFAULT_TTS_OUTPUT, get_tts_model, synthesize, synthesize_to_file

GREETING = "Assalam o alaikum, aap ka kya order hai?"
LONG_TEXT = (
    "Assalam o alaikum, Kasur Kitchen mein khush amdeed. "
    "Aaj aap kya order karna chahte hain? "
    "Humare paas chicken karahi, biryani aur pizza available hain. "
    "Delivery address bata dein taake order confirm ho sake."
)


def _sample_order() -> Order:
    order = Order()
    add_item_to_order(order, "chicken-karahi", "full", 2)
    order.delivery_address = "Model Town, Kasur"
    confirm_order(order)
    return order


def _is_riff_wav(wav_bytes: bytes) -> bool:
    return len(wav_bytes) >= 4 and wav_bytes[:4] == b"RIFF"


def test_greeting() -> None:
    print("Test 4.1: greeting...")
    wav = synthesize(GREETING)
    if not wav:
        raise SystemExit("FAIL 4.1: empty WAV bytes")
    if not _is_riff_wav(wav):
        raise SystemExit("FAIL 4.1: missing RIFF header")
    print("PASS 4.1")


def test_confirmation() -> None:
    print("Test 4.2: order confirmation...")
    text = build_confirmation_roman(_sample_order())
    wav = synthesize(text)
    if not wav or not _is_riff_wav(wav):
        raise SystemExit("FAIL 4.2: invalid confirmation WAV")
    print("PASS 4.2")


def test_empty_text() -> None:
    print("Test 4.3: empty text...")
    try:
        synthesize("")
    except ValueError as exc:
        if "empty" not in str(exc).lower():
            raise SystemExit(f"FAIL 4.3: unexpected error: {exc}") from exc
    else:
        raise SystemExit("FAIL 4.3: expected ValueError for empty text")
    print("PASS 4.3")


def test_long_text() -> None:
    print("Test 4.4: long text...")
    short = synthesize(GREETING)
    long_wav = synthesize(LONG_TEXT)
    if len(long_wav) <= len(short):
        raise SystemExit("FAIL 4.4: long text WAV not longer than greeting")
    print("PASS 4.4")


def test_file_output() -> None:
    print("Test 4.5: file output...")
    out_path = synthesize_to_file(GREETING)
    if not out_path.is_file():
        raise SystemExit(f"FAIL 4.5: file missing at {out_path}")
    sr, data = scipy.io.wavfile.read(out_path)
    if data.size == 0:
        raise SystemExit("FAIL 4.5: empty audio data")
    if sr <= 0:
        raise SystemExit("FAIL 4.5: invalid sample rate")
    if out_path != DEFAULT_TTS_OUTPUT:
        raise SystemExit("FAIL 4.5: default path mismatch")
    print("PASS 4.5")


def test_latency() -> None:
    print("Test 4.6: latency...")
    start = time.perf_counter()
    synthesize(GREETING)
    elapsed = time.perf_counter() - start
    print(f"Latency: {elapsed:.1f}s")
    if elapsed > 8:
        print("WARN 4.6: synthesis took longer than 8s on CPU")
    print("PASS 4.6")


def main() -> None:
    print("Loading TTS model (first run may download ~1-2 min)...")
    get_tts_model()

    test_greeting()
    test_confirmation()
    test_empty_text()
    test_long_text()
    test_file_output()
    test_latency()
    print("\nAll Phase 4 tests passed.")


if __name__ == "__main__":
    main()
