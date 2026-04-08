"""voice/doa.py — Direction of Arrival estimation via GCC-PHAT.

Works with a 2-channel mic input (the ReSpeaker XVF3800 exposes 2 channels).
Feeds audio frames in via push(), reads smoothed angle via angle property.

Output:
    angle   float degrees, 0 = straight ahead, negative = left, positive = right
    None    when not enough speech energy to estimate

Physical setup assumption:
    The two mic channels are the left-most and right-most elements of the array.
    MIC_SEPARATION_M controls the assumed physical distance between them.
    Default 0.065 m (65 mm) — measure your array and set DOA_MIC_SEPARATION in .env
    if different.

Usage:
    doa = DOAEstimator(sample_rate=16000)
    doa.push(stereo_frame)     # numpy float32 (N, 2)
    angle = doa.angle          # float | None
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SPEED_OF_SOUND = 343.0  # m/s at ~20°C

# Physical separation between the two recorded mic channels (metres).
# Measure your array — override with DOA_MIC_SEPARATION=0.065 in .env
try:
    MIC_SEPARATION_M = float(os.environ.get("DOA_MIC_SEPARATION", "0.065"))
except ValueError:
    MIC_SEPARATION_M = 0.065

# Energy threshold — frames quieter than this are skipped (avoids estimating silence)
RMS_THRESHOLD = float(os.environ.get("DOA_RMS_THRESHOLD", "0.01"))

# Smoothing: lower = smoother but slower to react (0.05–0.3 is a good range)
SMOOTH_ALPHA = float(os.environ.get("DOA_SMOOTH_ALPHA", "0.15"))

# Minimum interval between terminal log lines (seconds)
LOG_INTERVAL_S = 0.5


def _gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, fs: int) -> float:
    """Return time delay (seconds) of sig_b relative to sig_a using GCC-PHAT."""
    n = len(sig_a) + len(sig_b)
    A = np.fft.rfft(sig_a, n=n)
    B = np.fft.rfft(sig_b, n=n)
    R = A * np.conj(B)
    denom = np.abs(R)
    R /= np.where(denom > 1e-10, denom, 1e-10)  # PHAT weighting
    cc = np.fft.irfft(R, n=n)

    # Only search within physically possible lag range
    max_lag = int(np.ceil(fs * MIC_SEPARATION_M / SPEED_OF_SOUND)) + 1
    # irfft output: positive lags in [0..max_lag], negative in [n-max_lag..n]
    cc_clipped = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    lag_samples = np.argmax(cc_clipped) - max_lag
    return lag_samples / fs


def _tdoa_to_angle(tdoa_s: float) -> float:
    """Convert TDOA (seconds) to angle (degrees). Clipped to [-90, 90]."""
    ratio = np.clip(tdoa_s * SPEED_OF_SOUND / MIC_SEPARATION_M, -1.0, 1.0)
    return float(np.degrees(np.arcsin(ratio)))


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles (handles wraparound)."""
    d = (a - b + 180) % 360 - 180
    return d


class DOAEstimator:
    """Push audio frames in, read smoothed angle out.

    Thread-safe: push() is called from the audio callback thread,
    angle is read from the main thread.
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        self._fs = sample_rate
        self._smoothed: float | None = None
        self._lock = threading.Lock()
        self._last_log = 0.0
        # Accumulate frames until we have enough for a reliable estimate
        self._buf: list[np.ndarray] = []
        self._buf_samples = 0
        # ~80 ms of audio per estimate — good balance of latency vs accuracy
        self._window_samples = int(sample_rate * 0.08)

    def push(self, frame: np.ndarray) -> None:
        """Feed a multichannel float32 frame (N, 2). Silently ignores non-2ch input."""
        if frame.ndim != 2 or frame.shape[1] < 2:
            return

        self._buf.append(frame)
        self._buf_samples += frame.shape[0]

        if self._buf_samples >= self._window_samples:
            block = np.concatenate(self._buf, axis=0)[:self._window_samples]
            self._buf = []
            self._buf_samples = 0
            self._process(block)

    def _process(self, block: np.ndarray) -> None:
        ch_a = block[:, 0].astype(np.float64)
        ch_b = block[:, 1].astype(np.float64)

        # Skip quiet frames — estimating DOA from noise is meaningless
        rms = float(np.sqrt(np.mean(ch_a ** 2 + ch_b ** 2) / 2))
        if rms < RMS_THRESHOLD:
            return

        try:
            tdoa = _gcc_phat(ch_a, ch_b, self._fs)
            angle = _tdoa_to_angle(tdoa)
        except Exception:
            return

        with self._lock:
            if self._smoothed is None:
                self._smoothed = angle
            else:
                diff = _angle_diff(angle, self._smoothed)
                self._smoothed = self._smoothed + SMOOTH_ALPHA * diff

        self._maybe_log(rms)

    def _maybe_log(self, rms: float) -> None:
        now = time.monotonic()
        if now - self._last_log < LOG_INTERVAL_S:
            return
        self._last_log = now
        angle = self.angle
        if angle is None:
            return
        direction = "ahead"
        if angle < -10:
            direction = f"LEFT  {abs(angle):.0f}°"
        elif angle > 10:
            direction = f"RIGHT {angle:.0f}°"
        else:
            direction = f"AHEAD {abs(angle):.0f}°"
        print(f"[DOA] {direction}  (rms={rms:.4f})")

    @property
    def angle(self) -> float | None:
        """Smoothed angle in degrees. None until first speech frame processed."""
        with self._lock:
            return self._smoothed
