from __future__ import annotations

import time

import numpy as np
import pytest

import config
import run_stt
from app import stt


class _Segment:
    text = " ہاں جی "


def test_whisper_device_and_compute_type_pass_through(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def __init__(self, model_name, *, device, compute_type):
            captured.update(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
            )

        def transcribe(self, *args, **kwargs):
            return iter([]), object()

    monkeypatch.setattr(stt, "_MODEL", None)
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
    monkeypatch.setattr(config, "WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setattr(stt, "WhisperModel", FakeModel)

    stt.get_model()
    assert captured == {
        "model_name": "small",
        "device": "cpu",
        "compute_type": "int8",
    }


def test_cuda_preflight_error_has_clear_message(monkeypatch) -> None:
    monkeypatch.setattr(stt, "_MODEL", None)
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(config, "WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setattr(stt.ctranslate2, "get_cuda_device_count", lambda: 0)

    with pytest.raises(stt.SttInitError) as exc:
        stt.get_model()

    message = str(exc.value)
    assert "Whisper CUDA failed to initialize" in message
    assert "device=cuda" in message
    assert "compute_type=float16" in message


def test_cuda_warmup_disables_vad_and_consumes_decode(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def __init__(self, model_name, *, device, compute_type):
            pass

        def transcribe(self, audio, **kwargs):
            captured["vad_filter"] = kwargs["vad_filter"]
            captured["samples"] = len(audio)

            def generate():
                yield _Segment()

            return generate(), object()

    monkeypatch.setattr(stt, "_MODEL", None)
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
    monkeypatch.setattr(stt, "WhisperModel", FakeModel)

    stt.get_model()

    assert captured == {
        "vad_filter": False,
        "samples": config.SAMPLE_RATE,
    }


def test_cuda_runtime_error_becomes_stt_init_error(monkeypatch) -> None:
    class FakeModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    monkeypatch.setattr(stt, "_MODEL", FakeModel())
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(config, "WHISPER_COMPUTE_TYPE", "float16")

    with pytest.raises(stt.SttInitError) as exc:
        stt.transcribe_audio(
            np.zeros(config.SAMPLE_RATE, dtype=np.float32),
            config.SAMPLE_RATE,
            roman_bias=False,
            latinize=False,
        )

    message = str(exc.value)
    assert "cublas64_12.dll" in message
    assert "CUDA 12" in message


def test_transcription_timer_consumes_lazy_segments(monkeypatch) -> None:
    class FakeModel:
        def transcribe(self, *args, **kwargs):
            def generate():
                time.sleep(0.03)
                yield _Segment()

            return generate(), object()

    monkeypatch.setattr(stt, "_MODEL", FakeModel())
    text, elapsed = stt.transcribe_audio_timed(
        np.zeros(config.SAMPLE_RATE, dtype=np.float32),
        config.SAMPLE_RATE,
        roman_bias=False,
        latinize=False,
    )
    assert text == "ہاں جی"
    assert elapsed >= 0.03


def test_run_stt_file_happy_path(monkeypatch, capsys) -> None:
    called = {}

    monkeypatch.setattr(run_stt, "preload_stt", lambda: called.setdefault("preload", True))
    monkeypatch.setattr(
        run_stt,
        "transcribe_file_timed",
        lambda path, **kwargs: ("ہاں جی", 0.25),
    )

    run_stt.run_file("clip.wav", latency=True)

    output = capsys.readouterr().out
    assert called["preload"] is True
    assert "Transcribing file: clip.wav" in output
    assert "Heard: ہاں جی" in output
    assert "transcription_seconds=0.250" in output


def test_run_stt_file_error_exits_cleanly(monkeypatch, capsys) -> None:
    def fail_file(*args, **kwargs):
        raise RuntimeError("Could not decode WAV file bad.wav")

    monkeypatch.setattr(run_stt.sys, "argv", ["run_stt.py", "--file", "bad.wav"])
    monkeypatch.setattr(run_stt, "preload_stt", lambda: None)
    monkeypatch.setattr(run_stt, "transcribe_file_timed", fail_file)

    with pytest.raises(SystemExit) as exc:
        run_stt.main()

    assert exc.value.code == 1
    assert "Error: Could not decode WAV file bad.wav" in capsys.readouterr().out
