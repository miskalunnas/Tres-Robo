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


# Kielen tunnistus: Whisper palauttaa "finnish", "english" jne. -> ISO 639-1.
_LANG_NAME_TO_CODE: dict[str, str] = {
    "finnish": "fi",
    "english": "en",
    "swedish": "sv",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "dutch": "nl",
    "portuguese": "pt",
    "russian": "ru",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
}


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str | None = None,
    prompt: str | None = None,
    return_language: bool = False,
) -> str | tuple[str, str]:
    """Send audio to OpenAI Whisper API. Returns (text, lang_code) if return_language else text.
    When language=None, Whisper auto-detects. When return_language=True, uses verbose_json."""
    if audio.size == 0:
        return ("", "") if return_language else ""
    wav_bytes = _float32_to_wav_bytes(audio, sample_rate)
    client = _client()
    file_like = io.BytesIO(wav_bytes)
    file_like.name = "audio.wav"
    kwargs: dict = {
        "model": "whisper-1",
        "file": file_like,
    }
    if prompt:
        kwargs["prompt"] = prompt[:1000]
    if language is not None:
        kwargs["language"] = language
    if return_language:
        kwargs["response_format"] = "verbose_json"
    response = client.audio.transcriptions.create(**kwargs)
    text = (response.text or "").strip()
    if return_language:
        raw = getattr(response, "language", None) or ""
        lang_name = (raw or "").strip().lower()
        if not lang_name:
            return text, ""  # Let conversation layer infer from text; don't assume Finnish
        if len(lang_name) == 2:
            return text, lang_name  # Already ISO 639-1 (e.g. "en", "fi")
        lang_code = _LANG_NAME_TO_CODE.get(lang_name, "")
        return text, lang_code
    return text
