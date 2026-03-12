"""PCM audio output for Gemini Live.

Gemini sends 24kHz 16-bit mono PCM chunks. This module plays them via sounddevice
in a background thread so the receive loop is never blocked.
"""

import queue
import threading

import numpy as np
import sounddevice as sd

# Gemini outputs 24kHz by default. Match this exactly.
SAMPLE_RATE = 24_000


class AudioPlayer:
    """Thread-safe streaming PCM player.

    Usage:
        player = AudioPlayer()
        player.play(pcm_bytes)   # non-blocking, queues for playback
        player.stop()            # flush and stop
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._play_loop, daemon=True, name="audio-out")
        self._thread.start()

    def play(self, pcm_bytes: bytes) -> None:
        """Non-blocking: enqueue PCM bytes for playback."""
        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass  # Drop oldest implicitly — prefer recency over completeness

    def is_busy(self) -> bool:
        """True if there are chunks still waiting to be played."""
        return not self._queue.empty()

    def stop(self) -> None:
        """Stop playback and flush the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        sd.stop()

    def shutdown(self) -> None:
        """Signal the play loop to exit."""
        self._queue.put(None)

    def _play_loop(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                break
            if not data:
                continue
            try:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, samplerate=self._sr, blocking=True)
            except Exception as exc:
                import sys
                print(f"[AudioOut] Playback error: {exc}", file=sys.stderr)
