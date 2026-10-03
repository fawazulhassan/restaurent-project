"""Phase 4 Piper HTTP TTS unit tests (no live server required)."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import scipy.io.wavfile

import config
from app import tts

URDU_GREETING = "السلام علیکم، آپ کیا آرڈر کرنا چاہیں گے؟"


def _wav_bytes(
    *,
    sample_rate: int = 22050,
    stereo: bool = False,
    dtype=np.int16,
) -> bytes:
    frames = max(100, sample_rate // 20)
    tone = np.linspace(-12000, 12000, frames, dtype=np.float32)
    if stereo:
        samples = np.column_stack((tone, tone / 2))
    else:
        samples = tone
    samples = samples.astype(dtype)
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, samples)
    return buffer.getvalue()


def _response(
    *,
    method: str,
    url: str,
    status_code: int = 200,
    content: bytes = b"",
    json_data: dict | None = None,
) -> httpx.Response:
    request = httpx.Request(method, url)
    if json_data is not None:
        return httpx.Response(
            status_code,
            json=json_data,
            request=request,
        )
    return httpx.Response(
        status_code,
        content=content,
        request=request,
    )


class PiperTtsTests(unittest.TestCase):
    def test_empty_text_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            tts.synthesize("   ")

    def test_health_check_logs_version_and_returns_none(self) -> None:
        response = _response(
            method="GET",
            url=f"{config.PIPER_TTS_URL}/info",
            json_data={
                "version": "1.6.0",
                "voice": {"name": config.PIPER_TTS_VOICE},
            },
        )
        output = io.StringIO()
        with patch("app.tts.httpx.get", return_value=response), contextlib.redirect_stdout(
            output
        ):
            result = tts.check_tts_service()

        self.assertIsNone(result)
        self.assertIn("version 1.6.0", output.getvalue())
        self.assertIn(config.PIPER_TTS_VOICE, output.getvalue())

    def test_health_check_invalid_json_raises_service_error(self) -> None:
        response = _response(
            method="GET",
            url=f"{config.PIPER_TTS_URL}/info",
            content=b"not-json",
        )
        with patch("app.tts.httpx.get", return_value=response):
            with self.assertRaisesRegex(tts.TTSServiceError, "invalid JSON"):
                tts.check_tts_service()

    def test_health_check_timeout_raises_service_error(self) -> None:
        request = httpx.Request("GET", f"{config.PIPER_TTS_URL}/info")
        error = httpx.ReadTimeout("timeout", request=request)
        with patch("app.tts.httpx.get", side_effect=error):
            with self.assertRaisesRegex(tts.TTSServiceError, "timed out"):
                tts.check_tts_service()

    def test_synthesize_sends_urdu_and_voice_and_returns_16khz_mono(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=_wav_bytes(),
        )
        with patch("app.tts.httpx.post", return_value=response) as post:
            wav = tts.synthesize(URDU_GREETING)

        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["json"]["text"], URDU_GREETING)
        self.assertEqual(kwargs["json"]["voice"], config.PIPER_TTS_VOICE)
        self.assertTrue(wav.startswith(b"RIFF"))
        sample_rate, samples = scipy.io.wavfile.read(io.BytesIO(wav))
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(samples.dtype, np.int16)

    def test_resampling_preserves_duration(self) -> None:
        source_wav = _wav_bytes()
        source_rate, source_samples = scipy.io.wavfile.read(io.BytesIO(source_wav))

        converted_wav = tts._convert_piper_wav(source_wav)
        target_rate, target_samples = scipy.io.wavfile.read(
            io.BytesIO(converted_wav)
        )

        source_duration = len(source_samples) / source_rate
        target_duration = len(target_samples) / target_rate
        self.assertAlmostEqual(source_duration, target_duration, places=3)

    def test_native_22050_mode_preserves_rate_and_duration(self) -> None:
        source_wav = _wav_bytes()
        source_rate, source_samples = scipy.io.wavfile.read(io.BytesIO(source_wav))

        with patch("app.tts.config.TTS_SAMPLE_RATE", 22050):
            converted_wav = tts._convert_piper_wav(source_wav)

        target_rate, target_samples = scipy.io.wavfile.read(
            io.BytesIO(converted_wav)
        )
        self.assertEqual(target_rate, 22050)
        self.assertEqual(len(target_samples), len(source_samples))
        self.assertEqual(source_rate, target_rate)

    def test_stereo_is_converted_to_mono(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=_wav_bytes(stereo=True),
        )
        with patch("app.tts.httpx.post", return_value=response):
            wav = tts.synthesize(URDU_GREETING)

        sample_rate, samples = scipy.io.wavfile.read(io.BytesIO(wav))
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.ndim, 1)

    def test_values_are_clipped_and_converted_to_pcm16(self) -> None:
        samples = np.array([-50000, -32768, 0, 32767, 50000], dtype=np.int32)
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, 16000, samples)

        wav = tts._convert_piper_wav(buffer.getvalue())
        _, converted = scipy.io.wavfile.read(io.BytesIO(wav))
        self.assertEqual(converted.dtype, np.int16)
        self.assertEqual(int(converted.min()), -32768)
        self.assertEqual(int(converted.max()), 32767)

    def test_non_mono_after_conversion_raises_service_error(self) -> None:
        samples = np.zeros((10, 2, 2), dtype=np.int16)
        with patch("app.tts._decode_wav", return_value=(16000, samples)):
            with self.assertRaisesRegex(tts.TTSServiceError, "not mono"):
                tts._convert_piper_wav(b"unused")

    def test_invalid_non_riff_response_raises_service_error(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=b"<html>rate limited</html>",
        )
        with patch("app.tts.httpx.post", return_value=response):
            with self.assertRaisesRegex(tts.TTSServiceError, "RIFF"):
                tts.synthesize(URDU_GREETING)

    def test_invalid_riff_wav_raises_service_error(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=b"RIFF\x00\x00\x00\x00WAVE",
        )
        with patch("app.tts.httpx.post", return_value=response):
            with self.assertRaisesRegex(tts.TTSServiceError, "could not be decoded"):
                tts.synthesize(URDU_GREETING)

    def test_synthesis_http_error_raises_service_error(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            status_code=503,
            content=b"unavailable",
        )
        with patch("app.tts.httpx.post", return_value=response):
            with self.assertRaisesRegex(tts.TTSServiceError, "HTTP 503"):
                tts.synthesize(URDU_GREETING)

    def test_unexpected_sample_rate_raises_service_error(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=_wav_bytes(sample_rate=44100),
        )
        with patch("app.tts.httpx.post", return_value=response):
            with self.assertRaisesRegex(tts.TTSServiceError, "sample rate 44100"):
                tts.synthesize(URDU_GREETING)

    def test_synthesize_to_file_writes_resampled_output(self) -> None:
        response = _response(
            method="POST",
            url=f"{config.PIPER_TTS_URL}/synthesize",
            content=_wav_bytes(),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "urdu.wav"
            with patch("app.tts.httpx.post", return_value=response):
                saved_path = tts.synthesize_to_file(URDU_GREETING, output_path)

            sample_rate, samples = scipy.io.wavfile.read(saved_path)
            self.assertEqual(saved_path, output_path)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(samples.ndim, 1)
            self.assertEqual(samples.dtype, np.int16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
