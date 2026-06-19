import io
import re
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import torch
from scipy.signal import resample
from transformers import AutoTokenizer, VitsModel, set_seed

import config

_TTS_MODEL = None
_TTS_TOKENIZER = None

DEFAULT_TTS_OUTPUT = Path(__file__).parent.parent / "data" / "tts" / "out.wav"


def get_tts_model() -> tuple[VitsModel, AutoTokenizer]:
    global _TTS_MODEL, _TTS_TOKENIZER
    if _TTS_MODEL is None:
        _TTS_TOKENIZER = AutoTokenizer.from_pretrained(config.TTS_MODEL)
        _TTS_MODEL = VitsModel.from_pretrained(config.TTS_MODEL)
        _TTS_MODEL.to(config.TTS_DEVICE)
        _TTS_MODEL.eval()
    return _TTS_MODEL, _TTS_TOKENIZER


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\n])\s+", text.strip())
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks or [text.strip()]


def _waveform_to_int16(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32).squeeze()
    if arr.size == 0:
        return np.array([], dtype=np.int16)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / peak
    return (arr * 32767).astype(np.int16)


def _resample_if_needed(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples
    num = int(len(samples) * target_rate / source_rate)
    return resample(samples, num).astype(np.int16)


def _synthesize_chunk(
    text: str, model: VitsModel, tokenizer: AutoTokenizer
) -> np.ndarray:
    set_seed(555)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(config.TTS_DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    waveform = output.waveform.squeeze().cpu().numpy()
    samples = _waveform_to_int16(waveform)
    return _resample_if_needed(
        samples, model.config.sampling_rate, config.TTS_SAMPLE_RATE
    )


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, samples)
    return buf.getvalue()


def synthesize(text: str) -> bytes:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("TTS text is empty")

    model, tokenizer = get_tts_model()
    chunks = _split_sentences(cleaned)

    if len(chunks) == 1:
        samples = _synthesize_chunk(chunks[0], model, tokenizer)
        return _samples_to_wav_bytes(samples, config.TTS_SAMPLE_RATE)

    gap = np.zeros(int(0.2 * config.TTS_SAMPLE_RATE), dtype=np.int16)
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        parts.append(_synthesize_chunk(chunk, model, tokenizer))
        if index < len(chunks) - 1:
            parts.append(gap)

    combined = np.concatenate(parts)
    return _samples_to_wav_bytes(combined, config.TTS_SAMPLE_RATE)


def synthesize_to_file(text: str, path: Path | str | None = None) -> Path:
    out_path = Path(path) if path is not None else DEFAULT_TTS_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(synthesize(text))
    return out_path


def play_audio(wav_bytes: bytes) -> None:
    sr, data = scipy.io.wavfile.read(io.BytesIO(wav_bytes))
    sd.play(data, samplerate=sr)
    sd.wait()
