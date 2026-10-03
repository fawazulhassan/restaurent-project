import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import stt


class FakeWhisperModel:
    def __init__(self) -> None:
        self.audio = None
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return (
            [
                SimpleNamespace(text=" السلام علیکم "),
                SimpleNamespace(text="کیا آرڈر کریں گے؟"),
            ],
            SimpleNamespace(language="ur"),
        )


class WhisperProviderTests(unittest.TestCase):
    def test_audio_uses_local_whisper_and_preserves_urdu(self) -> None:
        model = FakeWhisperModel()
        audio = np.zeros(16_000, dtype=np.float32)

        with (
            patch.object(stt.config, "STT_PROVIDER", "whisper"),
            patch.object(stt.config, "WHISPER_LANGUAGE", "ur"),
            patch.object(stt, "get_model", return_value=model),
        ):
            text = stt.transcribe_audio(
                audio, 16_000, roman_bias=False, latinize=False
            )

        self.assertEqual(text, "السلام علیکم کیا آرڈر کریں گے؟")
        self.assertEqual(model.audio.shape, (16_000,))
        self.assertEqual(model.kwargs["language"], "ur")
        self.assertTrue(model.kwargs["vad_filter"])

    def test_audio_is_resampled_to_16khz(self) -> None:
        model = FakeWhisperModel()
        audio = np.zeros(8_000, dtype=np.float32)

        with (
            patch.object(stt.config, "STT_PROVIDER", "whisper"),
            patch.object(stt, "get_model", return_value=model),
        ):
            stt.transcribe_audio(
                audio, 8_000, roman_bias=False, latinize=False
            )

        self.assertEqual(model.audio.shape, (16_000,))

    def test_preload_uses_whisper_model(self) -> None:
        with (
            patch.object(stt.config, "STT_PROVIDER", "whisper"),
            patch.object(stt, "get_model") as get_model,
        ):
            stt.preload_stt()

        get_model.assert_called_once_with()

    def test_unknown_provider_is_rejected(self) -> None:
        with patch.object(stt.config, "STT_PROVIDER", "unknown"):
            with self.assertRaisesRegex(
                RuntimeError, "Unsupported STT_PROVIDER"
            ):
                stt.preload_stt()

    def test_roman_bias_initial_prompt_contains_size_words(self) -> None:
        kwargs = stt._transcribe_kwargs(roman_bias=True)
        self.assertIn("initial_prompt", kwargs)
        self.assertIn("آدھی", kwargs["initial_prompt"])
        self.assertIn("پوری", kwargs["initial_prompt"])


if __name__ == "__main__":
    unittest.main()
