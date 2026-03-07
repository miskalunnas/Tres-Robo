"""Text-to-speech output via ElevenLabs.

Falls back to a print stub if the API key is not set.
Audio is played through the system default output (Bluetooth speaker).
"""

import os
import queue
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv()

_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# Lazy-load the ElevenLabs client only when needed.
_client = None


class _SpeechRequest:
    def __init__(self, text: str) -> None:
        self.text = text
        self.done = threading.Event()


class _TTSPlayer:
    def __init__(self) -> None:
        self._queue: "queue.Queue[_SpeechRequest]" = queue.Queue()
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._current_proc: subprocess.Popen | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text: str, *, block: bool = False) -> None:
        if not text:
            return
        print(f"[Robot] {text}")
        request = _SpeechRequest(text)
        self._queue.put(request)
        if block:
            request.done.wait()

    def is_speaking(self) -> bool:
        return self._speaking.is_set() or not self._queue.empty()

    def interrupt(self) -> None:
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            request.done.set()
            self._queue.task_done()

        with self._lock:
            proc = self._current_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            self._speaking.set()
            try:
                self._play_text(request.text)
            finally:
                request.done.set()
                self._speaking.clear()
                self._queue.task_done()

    def _play_text(self, text: str) -> None:
        if not _ELEVENLABS_API_KEY or not _VOICE_ID:
            return

        client = _get_client()
        if client is None:
            return

        try:
            t0 = time.monotonic()
            audio = client.text_to_speech.convert(
                voice_id=_VOICE_ID,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
            )
            # Stream MP3 to mpg123 via PulseAudio output (required for Bluetooth).
            proc = subprocess.Popen(
                ["mpg123", "-o", "pulse", "-q", "-"],
                stdin=subprocess.PIPE,
            )
            with self._lock:
                self._current_proc = proc
            first_chunk = True
            for chunk in audio:
                if first_chunk:
                    print(f"[Timing] TTS first-chunk: {time.monotonic()-t0:.2f}s")
                    first_chunk = False
                if proc.stdin is None:
                    break
                proc.stdin.write(chunk)
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait()
            print(f"[Timing] TTS total (API+playback): {time.monotonic()-t0:.2f}s")
        except Exception as exc:
            print(f"[TTS] Error: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._current_proc = None


def _get_client():
    global _client
    if _client is None:
        try:
            from elevenlabs.client import ElevenLabs
            _client = ElevenLabs(api_key=_ELEVENLABS_API_KEY)
        except ImportError:
            print("[TTS] elevenlabs package not installed. Run: pip install elevenlabs", file=sys.stderr)
    return _client


_player = _TTSPlayer()


def speak(text: str, *, block: bool = False) -> None:
    """Queue text for speech output."""
    _player.speak(text, block=block)


def is_speaking() -> bool:
    """Return True while speech is playing or queued."""
    return _player.is_speaking()


def interrupt() -> None:
    """Stop queued and current playback when possible."""
    _player.interrupt()
