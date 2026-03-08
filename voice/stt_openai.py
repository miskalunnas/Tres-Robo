"""OpenAI Whisper API for speech-to-text. Replaces local faster-whisper when enabled."""
import io
import os
import wave

import numpy as np

# OPENAI_API_KEY from env (e.g. .env via python-dotenv).
_openai_client = None


def _client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono [-1, 1] to 16-bit PCM WAV bytes."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())
    buf.seek(0)
    return buf.read()


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str = "fi",
    prompt: str | None = None,
) -> str:
    """Send audio to OpenAI Whisper API and return transcribed text."""
    if audio.size == 0:
        return ""
    wav_bytes = _float32_to_wav_bytes(audio, sample_rate)
    client = _client()
    # API expects a file-like object with name attribute for extension.
    file_like = io.BytesIO(wav_bytes)
    file_like.name = "audio.wav"
    kwargs = {
        "model": "whisper-1",
        "file": file_like,
        "language": language,
    }
    if prompt:
        kwargs["prompt"] = prompt[:1000]  # API limit
    response = client.audio.transcriptions.create(**kwargs)
    return (response.text or "").strip()
