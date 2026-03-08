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
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()

_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# Lazy-load the ElevenLabs client only when needed.
_client = None


class SpeechHandle:
    """Track the lifecycle of a single spoken reply."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.started = threading.Event()
        self.finished = threading.Event()
        self.interrupted = threading.Event()
        self._callbacks: list[Callable[["SpeechHandle"], None]] = []
        self._lock = threading.Lock()

    def wait(self, timeout: float | None = None) -> bool:
        return self.finished.wait(timeout=timeout)

    def add_done_callback(self, callback: Callable[["SpeechHandle"], None]) -> None:
        with self._lock:
            if self.finished.is_set():
                callback(self)
                return
            self._callbacks.append(callback)

    def _mark_started(self) -> None:
        self.started.set()

    def _mark_finished(self, *, interrupted: bool) -> None:
        callbacks: list[Callable[["SpeechHandle"], None]]
        with self._lock:
            if interrupted:
                self.interrupted.set()
            self.finished.set()
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception as exc:
                print(f"[TTS] Callback error: {exc}", file=sys.stderr)


class _SpeechRequest:
    def __init__(self, text: str, handle: SpeechHandle) -> None:
        self.text = text
        self.handle = handle


class _TTSPlayer:
    def __init__(self) -> None:
        self._queue: "queue.Queue[_SpeechRequest]" = queue.Queue()
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._busy = threading.Event()
        self._current_proc: subprocess.Popen | None = None
        self._current_handle: SpeechHandle | None = None
        self._interrupt_requested = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text: str, *, block: bool = False) -> SpeechHandle:
        handle = SpeechHandle(text)
        if not text:
            handle._mark_finished(interrupted=False)
            return handle
        print(f"[Robot] {text}")
        request = _SpeechRequest(text, handle)
        self._busy.set()
        self._queue.put(request)
        if block:
            handle.wait()
        return handle

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def interrupt(self) -> None:
        interrupted_any = False
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            interrupted_any = True
            request.handle._mark_finished(interrupted=True)
            self._queue.task_done()

        with self._lock:
            self._interrupt_requested = True
            proc = self._current_proc
            if proc is None and interrupted_any:
                self._busy.clear()
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            self._speaking.set()
            with self._lock:
                self._current_handle = request.handle
                self._interrupt_requested = False
            request.handle._mark_started()
            try:
                interrupted = self._play_text(request.text)
            finally:
                request.handle._mark_finished(interrupted=interrupted)
                self._speaking.clear()
                with self._lock:
                    self._current_handle = None
                    if self._queue.empty():
                        self._busy.clear()
                self._queue.task_done()

    def _play_text(self, text: str) -> bool:
        if not _ELEVENLABS_API_KEY or not _VOICE_ID:
            return False

        client = _get_client()
        if client is None:
            return False

        interrupted = False
        try:
            t0 = time.monotonic()
            # Use streaming API so we get first audio bytes earlier (lower latency).
            audio_stream = _get_audio_stream(client, text)
            if audio_stream is None:
                return False
            # Start mpg123 and feed chunks as they arrive.
            proc = subprocess.Popen(
                ["mpg123", "-o", "pulse", "-q", "-"],
                stdin=subprocess.PIPE,
            )
            with self._lock:
                self._current_proc = proc
            first_chunk = True
            for chunk in audio_stream:
                if self._was_interrupted():
                    interrupted = True
                    break
                if first_chunk:
                    print(f"[Timing] TTS first-chunk: {time.monotonic()-t0:.2f}s")
                    first_chunk = False
                if proc.stdin is None:
                    break
                if isinstance(chunk, bytes):
                    data = chunk
                else:
                    data = bytes(chunk) if not isinstance(chunk, str) else chunk.encode()
                if not data:
                    continue
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    interrupted = True
                    break
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    interrupted = True
            proc.wait()
            print(f"[Timing] TTS total (API+playback): {time.monotonic()-t0:.2f}s")
        except Exception as exc:
            print(f"[TTS] Error: {exc}", file=sys.stderr)
            interrupted = interrupted or self._was_interrupted()
        finally:
            with self._lock:
                interrupted = interrupted or self._interrupt_requested
                self._current_proc = None
                self._interrupt_requested = False
        return interrupted

    def _was_interrupted(self) -> bool:
        with self._lock:
            return self._interrupt_requested


def _get_client():
    global _client
    if _client is None:
        try:
            from elevenlabs.client import ElevenLabs
            _client = ElevenLabs(api_key=_ELEVENLABS_API_KEY)
        except ImportError:
            print("[TTS] elevenlabs package not installed. Run: pip install elevenlabs", file=sys.stderr)
    return _client


def _get_audio_stream(client, text: str):
    """Return an iterator of audio bytes (streaming preferred for lower first-byte latency)."""
    kwargs = {
        "voice_id": _VOICE_ID,
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "output_format": "mp3_44100_128",
    }
    # Prefer streaming API: server sends chunks as generated so we get first bytes earlier.
    stream_method = getattr(
        getattr(client, "text_to_speech", None),
        "convert_as_stream",
        None,
    ) or getattr(
        getattr(client, "text_to_speech", None),
        "stream",
        None,
    )
    if stream_method is not None:
        try:
            stream = stream_method(**kwargs)
            if stream is not None:
                return stream
        except (TypeError, Exception):
            pass
    # Fallback: non-streaming convert (may return iterator of chunks or single bytes).
    try:
        result = client.text_to_speech.convert(**kwargs)
        if result is None:
            return None
        # If it's bytes-like, treat as single chunk; otherwise assume it's an iterator.
        if isinstance(result, (bytes, bytearray)):
            return iter([bytes(result)])
        return result
    except Exception:
        return None


_player = _TTSPlayer()


def speak(text: str, *, block: bool = False) -> SpeechHandle:
    """Queue text for speech output and return its lifecycle handle."""
    return _player.speak(text, block=block)


def is_speaking() -> bool:
    """Return True while speech audio is actively playing."""
    return _player.is_speaking()


def is_busy() -> bool:
    """Return True while speech is playing or additional replies are queued."""
    return _player.is_busy()


def interrupt() -> None:
    """Stop queued and current playback when possible."""
    _player.interrupt()
