from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import config
from app.romanize import to_roman_latin

_MODEL: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _MODEL


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr[:, 0]
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32, copy=False)


def _transcribe_kwargs(*, roman_bias: bool = False) -> dict:
    kwargs: dict = {
        "beam_size": config.WHISPER_BEAM_SIZE,
        "condition_on_previous_text": False,
    }
    if roman_bias:
        # Roman Urdu speech → Latin script (for run_voice.py)
        kwargs["language"] = None
        kwargs["initial_prompt"] = config.WHISPER_ROMAN_PROMPT
    else:
        kwargs["language"] = config.WHISPER_LANGUAGE
        if config.WHISPER_INITIAL_PROMPT:
            kwargs["initial_prompt"] = config.WHISPER_INITIAL_PROMPT
    return kwargs


def _finalize_transcript(text: str, *, latinize: bool) -> str:
    if not text or not latinize:
        return text
    return to_roman_latin(text)


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    roman_bias: bool = True,
    latinize: bool | None = None,
) -> str:
    if latinize is None:
        latinize = roman_bias
    del sample_rate  # faster-whisper resamples internally when needed
    arr = _normalize_audio(audio)
    if arr.size == 0:
        return ""

    model = get_model()
    segments, _ = model.transcribe(arr, **_transcribe_kwargs(roman_bias=roman_bias))
    text = " ".join(seg.text for seg in segments).strip()
    return _finalize_transcript(text, latinize=latinize)


def transcribe_file(
    wav_path: str | Path,
    *,
    roman_bias: bool = True,
    latinize: bool | None = None,
) -> str:
    if latinize is None:
        latinize = roman_bias
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    model = get_model()
    segments, _ = model.transcribe(
        str(path), **_transcribe_kwargs(roman_bias=roman_bias)
    )
    text = " ".join(seg.text for seg in segments).strip()
    return _finalize_transcript(text, latinize=latinize)


def record_and_transcribe(
    duration_seconds: float | None = None,
    *,
    quiet: bool = False,
    roman_bias: bool = True,
    latinize: bool | None = None,
) -> str:
    duration = (
        config.DEFAULT_RECORD_SECONDS
        if duration_seconds is None
        else duration_seconds
    )
    frames = int(duration * config.SAMPLE_RATE)

    try:
        audio = sd.rec(
            frames,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except sd.PortAudioError:
        raise RuntimeError(
            "Microphone not accessible — check Windows Privacy settings."
        ) from None

    arr = _normalize_audio(audio)
    if not quiet:
        print(f"Recorded {duration:.1f}s ({arr.size} samples)")
    return transcribe_audio(
        arr, config.SAMPLE_RATE, roman_bias=roman_bias, latinize=latinize
    )


if __name__ == "__main__":
    print("Loading Whisper model...")
    text = record_and_transcribe(5)
    if text:
        print(f"Heard: {text}")
    else:
        print("No speech detected.")
