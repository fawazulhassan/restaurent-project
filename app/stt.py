import io
import math
import os
import site
import time
from pathlib import Path

import ctranslate2
import httpx
import numpy as np
import scipy.io.wavfile
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

import config
from app.romanize import to_roman_latin

_MODEL: WhisperModel | None = None
_CUDA_DLL_HANDLES = []


class SttInitError(RuntimeError):
    pass


def _is_cuda_requested() -> bool:
    return config.WHISPER_DEVICE.strip().lower() != "cpu"


def _add_cuda_dll_directories() -> None:
    if not _is_cuda_requested() or os.name != "nt":
        return
    candidates = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
    ]
    for root in site.getsitepackages():
        packages = Path(root) / "nvidia"
        candidates.extend(
            [
                packages / "cublas" / "bin",
                packages / "cudnn" / "bin",
                packages / "cuda_nvrtc" / "bin",
            ]
        )
    for path in candidates:
        if not path.is_dir():
            continue
        path_str = str(path)
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(path_str))
            except OSError:
                pass


def _cuda_error_message(exc: Exception | str) -> str:
    detail = str(exc)
    return (
        "Whisper CUDA failed to initialize with "
        f"device={config.WHISPER_DEVICE} "
        f"compute_type={config.WHISPER_COMPUTE_TYPE}. "
        "Install/repair CUDA cuBLAS/cuDNN compatible with this CTranslate2 "
        "version. CTranslate2 4.x typically needs CUDA 12 and cuDNN 9. "
        "For older CUDA 11 setups, use ctranslate2==3.24.0, or switch "
        "WHISPER_DEVICE=cpu and WHISPER_COMPUTE_TYPE=int8."
        f" Details: {detail}"
    )


def _preflight_cuda() -> None:
    if not _is_cuda_requested():
        return
    _add_cuda_dll_directories()
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            raise SttInitError(_cuda_error_message("no CUDA device found"))
        supported = ctranslate2.get_supported_compute_types("cuda")
    except SttInitError:
        raise
    except Exception as exc:
        raise SttInitError(_cuda_error_message(exc)) from exc
    if config.WHISPER_COMPUTE_TYPE not in supported:
        raise SttInitError(
            _cuda_error_message(
                f"compute type {config.WHISPER_COMPUTE_TYPE!r} not in {supported}"
            )
        )


def _consume_segments(segments) -> list:
    return list(segments)


def _warm_up_model(model: WhisperModel) -> None:
    samples = np.arange(config.SAMPLE_RATE, dtype=np.float32)
    warmup_audio = 0.001 * np.sin(2 * np.pi * 440.0 * samples / config.SAMPLE_RATE)
    kwargs = _transcribe_kwargs(roman_bias=False)
    kwargs["vad_filter"] = False
    segments, _ = model.transcribe(
        warmup_audio.astype(np.float32),
        **kwargs,
    )
    _consume_segments(segments)


def get_model() -> WhisperModel:
    """Load and cache the configured local faster-whisper model."""
    global _MODEL
    if _MODEL is None:
        try:
            _add_cuda_dll_directories()
            _preflight_cuda()
            _MODEL = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
            _warm_up_model(_MODEL)
        except SttInitError:
            raise
        except Exception as exc:
            if _is_cuda_requested():
                raise SttInitError(_cuda_error_message(exc)) from exc
            raise RuntimeError(
                f"Could not load faster-whisper model '{config.WHISPER_MODEL}': {exc}"
            ) from exc
    return _MODEL


def preload_stt() -> None:
    """Validate the configured STT provider and preload local Whisper."""
    if config.STT_PROVIDER == "whisper":
        print(f"Loading local Whisper STT ({config.WHISPER_MODEL})...")
        get_model()
        print(
            "Local Whisper STT ready "
            f"(device={config.WHISPER_DEVICE}, "
            f"compute_type={config.WHISPER_COMPUTE_TYPE})."
        )
        return
    if config.STT_PROVIDER == "elevenlabs":
        _require_elevenlabs_provider()
        print("ElevenLabs STT configured.")
        return
    raise RuntimeError(
        f"Unsupported STT_PROVIDER '{config.STT_PROVIDER}'. "
        "Use 'whisper' or 'elevenlabs'."
    )


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32, copy=False)


def _transcribe_kwargs(*, roman_bias: bool = False) -> dict:
    kwargs: dict = {
        "beam_size": config.WHISPER_BEAM_SIZE,
        "condition_on_previous_text": False,
        "vad_filter": True,
    }
    if config.WHISPER_LANGUAGE:
        kwargs["language"] = config.WHISPER_LANGUAGE
    prompt = (
        config.WHISPER_ROMAN_PROMPT
        if roman_bias
        else config.WHISPER_INITIAL_PROMPT
    )
    if prompt:
        kwargs["initial_prompt"] = prompt
    return kwargs


def _resample_for_whisper(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    arr = _normalize_audio(audio)
    if sample_rate == config.SAMPLE_RATE:
        return arr
    if sample_rate <= 0:
        raise RuntimeError(f"Invalid STT sample rate: {sample_rate}")
    divisor = math.gcd(config.SAMPLE_RATE, sample_rate)
    return resample_poly(
        arr,
        up=config.SAMPLE_RATE // divisor,
        down=sample_rate // divisor,
    ).astype(np.float32)


def _whisper_transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    roman_bias: bool,
) -> str:
    arr = _resample_for_whisper(audio, sample_rate)
    try:
        segments, _ = get_model().transcribe(
            arr,
            **_transcribe_kwargs(roman_bias=roman_bias),
        )
        materialized = _consume_segments(segments)
        return " ".join(segment.text.strip() for segment in materialized).strip()
    except RuntimeError as exc:
        if _is_cuda_requested():
            raise SttInitError(_cuda_error_message(exc)) from exc
        raise
    except Exception as exc:
        raise RuntimeError(f"Local Whisper STT failed: {exc}") from exc


def _read_wav_for_transcription(path: Path) -> tuple[np.ndarray, int]:
    try:
        sample_rate, audio = scipy.io.wavfile.read(path)
    except Exception as exc:
        raise RuntimeError(f"Could not decode WAV file {path}: {exc}") from exc
    return _resample_for_whisper(audio, int(sample_rate)), config.SAMPLE_RATE


def _whisper_transcribe_file(path: Path, *, roman_bias: bool) -> str:
    try:
        audio, sample_rate = _read_wav_for_transcription(path)
        return _whisper_transcribe_audio(
            audio,
            sample_rate,
            roman_bias=roman_bias,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Local Whisper STT failed: {exc}") from exc


def _require_elevenlabs_provider() -> None:
    if config.STT_PROVIDER != "elevenlabs":
        raise RuntimeError(
            f"Unsupported STT_PROVIDER '{config.STT_PROVIDER}'. "
            "ElevenLabs transcription requires STT_PROVIDER=elevenlabs."
        )
    if not config.ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is missing. Copy .env.example to .env and set your key."
        )


def _build_stt_form_data() -> tuple[dict[str, str], dict[str, str]]:
    headers = {"xi-api-key": config.ELEVENLABS_API_KEY}
    data = {"model_id": config.ELEVENLABS_STT_MODEL}
    if config.ELEVENLABS_STT_LANGUAGE:
        data["language_code"] = config.ELEVENLABS_STT_LANGUAGE
    return headers, data


def _parse_transcript_response(payload: dict) -> str:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            "ElevenLabs STT response did not include usable 'text'. "
            "Check model_id/language_code or API response format."
        )
    return text.strip()


def _elevenlabs_transcribe_bytes(
    audio_bytes: bytes, filename: str, content_type: str = "audio/wav"
) -> str:
    _require_elevenlabs_provider()
    headers, data = _build_stt_form_data()
    files = {"file": (filename, audio_bytes, content_type)}

    try:
        with httpx.Client(timeout=config.ELEVENLABS_STT_TIMEOUT_SECONDS) as client:
            response = client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers,
                data=data,
                files=files,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"ElevenLabs STT request timed out after "
            f"{config.ELEVENLABS_STT_TIMEOUT_SECONDS:.0f}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"ElevenLabs STT request failed: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError("ElevenLabs STT unauthorized: invalid API key.")
    if response.status_code in (402, 403, 429):
        raise RuntimeError(
            f"ElevenLabs STT quota/rate limit issue (HTTP {response.status_code})."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"ElevenLabs STT error HTTP {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("ElevenLabs STT returned non-JSON response.") from exc

    return _parse_transcript_response(payload)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, _normalize_audio(audio))
    return buf.getvalue()


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
    arr = _normalize_audio(audio)
    if arr.size == 0:
        return ""
    if config.STT_PROVIDER == "whisper":
        text = _whisper_transcribe_audio(
            arr,
            sample_rate,
            roman_bias=roman_bias,
        )
    elif config.STT_PROVIDER == "elevenlabs":
        wav_bytes = _audio_to_wav_bytes(arr, sample_rate)
        text = _elevenlabs_transcribe_bytes(wav_bytes, filename="audio.wav")
    else:
        raise RuntimeError(
            f"Unsupported STT_PROVIDER '{config.STT_PROVIDER}'. "
            "Use 'whisper' or 'elevenlabs'."
        )
    return _finalize_transcript(text, latinize=latinize)


def transcribe_audio_timed(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    roman_bias: bool = True,
    latinize: bool | None = None,
) -> tuple[str, float]:
    start = time.perf_counter()
    text = transcribe_audio(
        audio,
        sample_rate,
        roman_bias=roman_bias,
        latinize=latinize,
    )
    return text, time.perf_counter() - start


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
    if config.STT_PROVIDER == "whisper":
        text = _whisper_transcribe_file(path, roman_bias=roman_bias)
    elif config.STT_PROVIDER == "elevenlabs":
        text = _elevenlabs_transcribe_bytes(path.read_bytes(), filename=path.name)
    else:
        raise RuntimeError(
            f"Unsupported STT_PROVIDER '{config.STT_PROVIDER}'. "
            "Use 'whisper' or 'elevenlabs'."
        )
    return _finalize_transcript(text, latinize=latinize)


def transcribe_file_timed(
    wav_path: str | Path,
    *,
    roman_bias: bool = True,
    latinize: bool | None = None,
) -> tuple[str, float]:
    start = time.perf_counter()
    text = transcribe_file(wav_path, roman_bias=roman_bias, latinize=latinize)
    return text, time.perf_counter() - start


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


def record_clip(path: str | Path, seconds: float, *, force: bool = False) -> Path:
    out = Path(path)
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing clip: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * config.SAMPLE_RATE)
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
    scipy.io.wavfile.write(out, config.SAMPLE_RATE, _normalize_audio(audio))
    return out


def record_push_to_talk(
    *,
    roman_bias: bool = True,
    latinize: bool | None = None,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> str:
    """Press Enter to start (caller), Enter again to stop; transcribe captured audio."""
    min_sec = config.PTT_MIN_SECONDS if min_seconds is None else min_seconds
    max_sec = config.PTT_MAX_SECONDS if max_seconds is None else max_seconds
    max_samples = int(max_sec * config.SAMPLE_RATE)

    chunks: list[np.ndarray] = []
    total_samples = 0

    def callback(indata, frames, time_info, status):
        nonlocal total_samples
        if total_samples >= max_samples:
            return
        chunk = indata.copy()
        remaining = max_samples - total_samples
        if chunk.shape[0] > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total_samples += chunk.shape[0]

    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            input("Press Enter to stop...")
    except sd.PortAudioError:
        raise RuntimeError(
            "Microphone not accessible — check Windows Privacy settings."
        ) from None

    if not chunks:
        return ""

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = audio.size / config.SAMPLE_RATE
    if duration < min_sec:
        return ""

    return transcribe_audio(
        audio,
        config.SAMPLE_RATE,
        roman_bias=roman_bias,
        latinize=latinize,
    )


if __name__ == "__main__":
    print(f"Recording and transcribing with {config.STT_PROVIDER} STT...")
    text = record_and_transcribe(5)
    if text:
        print(f"Heard: {text}")
    else:
        print("No speech detected.")
