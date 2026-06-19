"""Phase 5 agent core — wires STT, dialog, and TTS into one turn."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from openai import RateLimitError

import config
from app.dialog import LLMRateLimitError, build_system_prompt, chat_turn
from app.order import Order, build_confirmation_roman, save_order
from app.stt import transcribe_audio
from app.tts import play_audio, synthesize


def preload_models() -> None:
    print("Loading models...")
    from app.stt import get_model
    from app.tts import get_tts_model

    get_model()
    get_tts_model()
    print("Ready.")


def reply_text_for_tts(reply: str, order: Order, is_complete: bool) -> str:
    if is_complete:
        return build_confirmation_roman(order)
    return reply.strip()


def process_text_turn(
    user_message: str,
    order: Order,
    messages: list[dict],
    *,
    log_latency: bool | None = None,
) -> tuple[str, bytes, Order, list[dict], bool]:
    """Returns (reply_text, reply_audio_wav, order, messages, is_complete)."""
    should_log = config.AGENT_LOG_LATENCY if log_latency is None else log_latency
    t0 = time.perf_counter()

    reply_text, order, messages, is_complete = chat_turn(
        user_message, order, messages
    )
    t_llm = time.perf_counter()

    tts_text = reply_text_for_tts(reply_text, order, is_complete)
    reply_audio = b""
    if tts_text:
        try:
            reply_audio = synthesize(tts_text)
        except ValueError:
            pass
    t_tts = time.perf_counter()

    if should_log:
        print(
            f"  [latency] llm={t_llm - t0:.1f}s tts={t_tts - t_llm:.1f}s "
            f"total={t_tts - t0:.1f}s"
        )

    return reply_text, reply_audio, order, messages, is_complete


def process_audio_turn(
    audio: np.ndarray,
    sample_rate: int,
    order: Order,
    messages: list[dict],
    *,
    log_latency: bool | None = None,
) -> tuple[str, str, bytes, Order, list[dict], bool]:
    """Returns (user_text, reply_text, reply_audio, order, messages, is_complete)."""
    should_log = config.AGENT_LOG_LATENCY if log_latency is None else log_latency
    t0 = time.perf_counter()

    user_text = transcribe_audio(audio, sample_rate)
    t_stt = time.perf_counter()

    reply_text, reply_audio, order, messages, is_complete = process_text_turn(
        user_text, order, messages, log_latency=False
    )
    t_end = time.perf_counter()

    if should_log:
        print(
            f"  [latency] stt={t_stt - t0:.1f}s llm+tts={t_end - t_stt:.1f}s "
            f"total={t_end - t0:.1f}s"
        )

    return user_text, reply_text, reply_audio, order, messages, is_complete


def play_greeting(greeting: str | None = None) -> None:
    text = greeting if greeting is not None else config.AGENT_GREETING
    if not text.strip():
        return
    try:
        play_audio(synthesize(text))
    except ValueError:
        pass


def run_conversation(
    record_fn: Callable[..., str],
    *,
    greeting: str | None = None,
    play_greeting_audio: bool = True,
    log_latency: bool = False,
    on_turn: Callable[..., Any] | None = None,
) -> Order | None:
    """Multi-turn session until order confirmed. Returns saved order or None."""
    preload_models()

    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]

    if play_greeting_audio:
        play_greeting(greeting)

    while True:
        try:
            user_text = record_fn().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return None
        except RuntimeError as e:
            print(f"Mic error: {e}")
            continue

        if not user_text:
            print("No speech detected. Try again.\n")
            continue

        try:
            reply_text, reply_audio, order, messages, is_complete = process_text_turn(
                user_text, order, messages, log_latency=log_latency
            )
        except (LLMRateLimitError, RateLimitError) as e:
            print(f"\nRate limit: {e}\n")
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        if on_turn is not None:
            on_turn(user_text, reply_text, reply_audio, order, is_complete)
        else:
            print(f"You: {user_text}")
            print(f"\nAI: {reply_text}\n")
            if reply_audio:
                play_audio(reply_audio)

        if is_complete:
            path = save_order(order)
            print(f"Order saved: {path}")
            return order

    return None
