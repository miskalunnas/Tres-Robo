"""Text-to-speech output via ElevenLabs.

Falls back to a print stub if the API key is not set.
Audio is played through the system default output (Bluetooth speaker).
"""
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# Lazy-load the ElevenLabs client only when needed.
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from elevenlabs.client import ElevenLabs
            _client = ElevenLabs(api_key=_ELEVENLABS_API_KEY)
        except ImportError:
            print("[TTS] elevenlabs package not installed. Run: pip install elevenlabs", file=sys.stderr)
    return _client


def speak(text: str) -> None:
    """Speak text aloud via ElevenLabs. Blocks until playback is complete."""
    print(f"[Robot] {text}")

    if not _ELEVENLABS_API_KEY or not _VOICE_ID:
        return  # stub mode — just the print above

    client = _get_client()
    if client is None:
        return

    try:
        t0 = time.monotonic()
        # pcm_16000: raw signed-16-bit mono at 16 kHz — no MP3 decode overhead,
        # smaller per-chunk size, plays directly via aplay.
        audio = client.text_to_speech.convert(
            voice_id=_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_16000",
        )
        # Stream raw PCM to paplay (PulseAudio) — reaches Bluetooth speaker.
        proc = subprocess.Popen(
            ["paplay", "--raw", "--rate=16000", "--channels=1", "--format=s16le", "-"],
            stdin=subprocess.PIPE,
        )
        first_chunk = True
        for chunk in audio:
            if first_chunk:
                print(f"[Timing] TTS first-chunk: {time.monotonic()-t0:.2f}s")
                first_chunk = False
            proc.stdin.write(chunk)
        proc.stdin.close()
        proc.wait()
        print(f"[Timing] TTS total (API+playback): {time.monotonic()-t0:.2f}s")
    except Exception as exc:
        print(f"[TTS] Error: {exc}", file=sys.stderr)
