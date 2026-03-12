"""PCM audio output for Gemini Live.

Gemini sends 24kHz 16-bit mono PCM chunks. This module plays them through
PulseAudio (paplay/aplay) so they route to the Bluetooth speaker — same
audio path as the existing ElevenLabs/mpg123 pipeline.

Falls back to sounddevice if paplay/aplay are not available.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading

import numpy as np

# Gemini outputs 24kHz 16-bit PCM mono by default.
SAMPLE_RATE = 24_000


def _find_player() -> str:
    """Detect which raw-PCM player is available."""
    # paplay (PulseAudio) — routes to BT speaker via default PulseAudio sink
    if shutil.which("paplay"):
        return "paplay"
    # aplay (ALSA) — works on Linux without PulseAudio
    if shutil.which("aplay"):
        return "aplay"
    # Fallback: sounddevice (may not route to BT speaker)
    return "sounddevice"


class AudioPlayer:
    """Thread-safe streaming PCM player.

    Streams 24kHz 16-bit mono PCM chunks. Uses a persistent subprocess
    (paplay/aplay) to avoid per-chunk process overhead.

    Usage:
        player = AudioPlayer()
        player.play(pcm_bytes)   # non-blocking
        player.is_busy()
        player.stop()            # flush, restart subprocess
        player.shutdown()        # stop the play loop
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self._player_type = _find_player()
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._play_loop, daemon=True, name="audio-out")
        self._thread.start()
        print(f"[AudioOut] Using: {self._player_type} ({self._sr} Hz)")

    def play(self, pcm_bytes: bytes) -> None:
        """Non-blocking: enqueue PCM bytes for playback."""
        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Drop if full — prefer latest audio
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_bytes)
            except queue.Empty:
                pass

    def is_busy(self) -> bool:
        """True if there are chunks waiting to be played."""
        return not self._queue.empty()

    def stop(self) -> None:
        """Flush pending audio and kill the current player process."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._kill_proc()

    def shutdown(self) -> None:
        """Signal the play loop to exit."""
        self.stop()
        self._queue.put(None)

    def _kill_proc(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    pass
                self._proc = None

    def _ensure_proc(self) -> subprocess.Popen | None:
        """Start (or restart) the player subprocess."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return self._proc

        if self._player_type == "paplay":
            cmd = [
                "paplay",
                f"--rate={self._sr}",
                "--format=s16le",
                "--channels=1",
                "--raw",
            ]
        elif self._player_type == "aplay":
            cmd = [
                "aplay",
                "-f", "S16_LE",
                "-r", str(self._sr),
                "-c", "1",
                "-q",
                "-",
            ]
        else:
            return None  # sounddevice fallback

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            with self._lock:
                self._proc = proc
            return proc
        except Exception as exc:
            print(f"[AudioOut] Failed to start {cmd[0]}: {exc}", file=sys.stderr)
            return None

    def _play_loop(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                break
            if not data:
                continue

            if self._player_type == "sounddevice":
                self._play_sounddevice(data)
            else:
                self._play_subprocess(data)

    def _play_subprocess(self, data: bytes) -> None:
        """Write PCM bytes to the paplay/aplay subprocess stdin."""
        proc = self._ensure_proc()
        if proc is None:
            self._play_sounddevice(data)  # fallback
            return
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # Process died, restart on next chunk
            with self._lock:
                self._proc = None

    def _play_sounddevice(self, data: bytes) -> None:
        """Fallback: play via sounddevice (may not route to BT speaker)."""
        try:
            import sounddevice as sd
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=self._sr, blocking=True)
        except Exception as exc:
            print(f"[AudioOut] sounddevice error: {exc}", file=sys.stderr)
