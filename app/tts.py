"""Piper HTTP text-to-speech client with telephony-ready WAV output."""

from __future__ import annotations

import io
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy.io.wavfile
import sounddevice as sd
from scipy.signal import resample_poly

import config

DEFAULT_TTS_OUTPUT = Path(__file__).parent.parent / "data" / "tts" / "out.wav"
PIPER_SOURCE_SAMPLE_RATE = 22050


class TTSServiceError(RuntimeError):
    """Raised when the Piper HTTP service cannot provide valid audio."""


def _endpoint(path: str) -> str:
    return f"{config.PIPER_TTS_URL.rstrip('/')}/{path.lstrip('/')}"


def _require_piper_provider() -> None:
    if config.TTS_PROVIDER != "piper_http":
        raise TTSServiceError(
            f"Unsupported TTS_PROVIDER '{config.TTS_PROVIDER}'. "
            "Only 'piper_http' is active."
        )


def _installed_piper_version() -> str:
    try:
        return version("piper-tts")
    except PackageNotFoundError:
        return "unknown"


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TTSServiceError("Piper /info returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise TTSServiceError("Piper /info returned an invalid JSON object.")
    return payload


def check_tts_service() -> None:
    """Check Piper readiness, log its version/voice, and return None."""
    _require_piper_provider()
    try:
        response = httpx.get(
            _endpoint("/info"),
            timeout=config.PIPER_TTS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise TTSServiceError(
            f"Piper health check timed out after "
            f"{config.PIPER_TTS_TIMEOUT_SECONDS:g}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise TTSServiceError(
            f"Piper health check failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise TTSServiceError(
            f"Piper TTS is unavailable at {config.PIPER_TTS_URL}: {exc}"
        ) from exc

    payload = _response_json(response)
    voice = payload.get("voice")
    voice_name = (
        str(voice.get("name", "unknown"))
        if isinstance(voice, dict)
        else "unknown"
    )
    reported_version = payload.get("version") or payload.get("piper_version")
    piper_version = (
        str(reported_version) if reported_version else _installed_piper_version()
    )
    print(f"Piper TTS ready (version {piper_version}, voice {voice_name}).")
    return None


def _decode_wav(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF":
        raise TTSServiceError("Piper returned invalid WAV data (missing RIFF header).")
    try:
        source_rate, samples = scipy.io.wavfile.read(io.BytesIO(wav_bytes))
    except Exception as exc:
        raise TTSServiceError("Piper returned WAV data that could not be decoded.") from exc
    if not isinstance(samples, np.ndarray) or samples.size == 0:
        raise TTSServiceError("Piper returned empty audio.")
    return int(source_rate), samples


def _convert_piper_wav(wav_bytes: bytes) -> bytes:
    source_rate, samples = _decode_wav(wav_bytes)

    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    if source_rate == config.TTS_SAMPLE_RATE:
        pass
    elif (
        source_rate == PIPER_SOURCE_SAMPLE_RATE
        and config.TTS_SAMPLE_RATE == 16000
    ):
        # 16,000 / 22,050 reduces to 320 / 441.
        samples = resample_poly(samples, up=320, down=441)
    else:
        raise TTSServiceError(
            f"Unsupported Piper sample rate {source_rate}; expected "
            f"the configured {config.TTS_SAMPLE_RATE} Hz output or "
            f"{PIPER_SOURCE_SAMPLE_RATE} Hz with a 16,000 Hz target."
        )

    # Normalize to float32 regardless of input path.
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # Values from int16 WAV are in [-32768, 32767] range.
    pcm16_samples = np.clip(samples, -32768, 32767).astype(np.int16)

    # Ensure 1-D.
    if pcm16_samples.ndim != 1:
        raise TTSServiceError("Audio is not mono after conversion")

    output_buffer = io.BytesIO()
    scipy.io.wavfile.write(
        output_buffer,
        config.TTS_SAMPLE_RATE,
        pcm16_samples,
    )
    return output_buffer.getvalue()


def synthesize(text: str) -> bytes:
    """Synthesize Urdu text with Piper and return 16 kHz mono PCM16 WAV bytes."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("TTS text is empty")

    _require_piper_provider()
    try:
        response = httpx.post(
            _endpoint("/synthesize"),
            json={
                "text": cleaned,
                "voice": config.PIPER_TTS_VOICE,
            },
            timeout=config.PIPER_TTS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise TTSServiceError(
            f"Piper synthesis timed out after "
            f"{config.PIPER_TTS_TIMEOUT_SECONDS:g}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise TTSServiceError(
            f"Piper synthesis failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise TTSServiceError(
            f"Piper synthesis request failed at {config.PIPER_TTS_URL}: {exc}"
        ) from exc

    return _convert_piper_wav(response.content)


def synthesize_to_file(text: str, path: Path | str | None = None) -> Path:
    """Save the resampled 16 kHz output returned by synthesize()."""
    out_path = Path(path) if path is not None else DEFAULT_TTS_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(synthesize(text))
    return out_path


def play_audio(wav_bytes: bytes) -> None:
    sample_rate, data = scipy.io.wavfile.read(io.BytesIO(wav_bytes))
    sd.play(data, samplerate=sample_rate)
    sd.wait()
