"""Text-to-speech: ElevenLabs tai Google Translate (gTTS).

Oletus: ElevenLabs (TTS_PROVIDER=elevenlabs). Vaihda TTS_PROVIDER=google ilmaiseen Google TTS:ään.
Vaatii ELEVENLABS_API_KEY ja ELEVENLABS_VOICE_ID .env:ssä.
"""

import io
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").strip().lower()
_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# Lazy-load ElevenLabs client only when needed.
_client = None


def _infer_tts_lang(text: str) -> str:
    """Yksinkertainen arvio: ä/ö → suomi, muuten englanti."""
    if not text or len(text) < 2:
        return "fi"
    t = text.lower()
    if "ä" in t or "ö" in t:
        return "fi"
    # Englanninkieliset yleissanat
    en_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "and", "or", "but"}
    words = set(t.split())
    if len(words & en_words) >= 2:
        return "en"
    return "fi"


def _get_playback_cmd() -> list[str] | None:
    """Palauttaa komennon joka lukee MP3 stdinistä. mpg123, ffplay tai mpv."""
    import shutil
    for cmd in ("mpg123", "ffplay", "mpv"):
        if shutil.which(cmd):
            if cmd == "mpg123":
                return ["mpg123", "-o", "pulse", "-q", "-"]
            if cmd == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"]
            if cmd == "mpv":
                return ["mpv", "--no-video", "--really-quiet", "-"]
    return None


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
        if TTS_PROVIDER == "google":
            return self._play_google(text)
        return self._play_elevenlabs(text)

    def _play_google(self, text: str) -> bool:
        """Google Translate TTS (gTTS) — ilmainen, ei API-avainta."""
        try:
            from gtts import gTTS
        except ImportError:
            print("[TTS] gTTS not installed. Run: pip install gTTS", file=sys.stderr)
            return False
        interrupted = False
        try:
            t0 = time.monotonic()
            lang = _infer_tts_lang(text)
            buf = io.BytesIO()
            tts = gTTS(text=text, lang=lang)
            tts.write_to_fp(buf)
            mp3_bytes = buf.getvalue()
            if not mp3_bytes:
                return False
            print(f"[Timing] TTS Google ({lang}): {time.monotonic()-t0:.2f}s")
            playback_cmd = _get_playback_cmd()
            if not playback_cmd:
                print("[TTS] No player (mpg123, ffplay or mpv). Install: sudo apt install mpg123", file=sys.stderr)
                return False
            proc = subprocess.Popen(
                playback_cmd,
                stdin=subprocess.PIPE,
            )
            with self._lock:
                self._current_proc = proc
            try:
                proc.stdin.write(mp3_bytes)
            except (BrokenPipeError, OSError):
                interrupted = True
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    interrupted = interrupted or True
            if not self._was_interrupted():
                proc.wait()
            else:
                interrupted = True
                try:
                    proc.terminate()
                    proc.wait()
                except Exception:
                    pass
            print(f"[Timing] TTS playback: {time.monotonic()-t0:.2f}s")
        except Exception as exc:
            print(f"[TTS] Google TTS error: {exc}", file=sys.stderr)
            interrupted = interrupted or self._was_interrupted()
        finally:
            with self._lock:
                interrupted = interrupted or self._interrupt_requested
                self._current_proc = None
                self._interrupt_requested = False
        return interrupted

    def _play_elevenlabs(self, text: str) -> bool:
        """ElevenLabs TTS — vaatii ELEVENLABS_API_KEY ja ELEVENLABS_VOICE_ID."""
        if not _ELEVENLABS_API_KEY or not _VOICE_ID:
            print("[TTS] ElevenLabs: set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env", file=sys.stderr)
            return False
        client = _get_client()
        if client is None:
            return False
        interrupted = False
        try:
            t0 = time.monotonic()
            audio_stream = _get_audio_stream(client, text)
            if audio_stream is None:
                return False
            playback_cmd = _get_playback_cmd()
            if not playback_cmd:
                print("[TTS] No player (mpg123, ffplay or mpv). Install: sudo apt install mpg123", file=sys.stderr)
                return False
            proc = subprocess.Popen(
                playback_cmd,
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
                data = bytes(chunk) if not isinstance(chunk, (bytes, bytearray)) else chunk
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

if TTS_PROVIDER == "google":
    print("[TTS] Using Google Translate (gTTS)")
else:
    print("[TTS] Using ElevenLabs (set TTS_PROVIDER=google for free Google TTS)")


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
