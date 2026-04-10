"""Pan/tilt servo control for Tres-Robo robot head.

Two SG90 micro servos (180°) connected to a separate 5 V power supply
(NOT the Raspberry Pi 5 V pin — servos draw too much current).

Wiring:
    Pan servo  signal → GPIO12 (Pin 32)
    Tilt servo signal → GPIO13 (Pin 33)
    Both servo GND    → common GND with Pi
    Both servo VCC    → external 5 V supply

SG90 PWM spec:
    Frequency : 50 Hz  (20 ms period)
    0°  pulse : 500 µs
    90° pulse : 1500 µs  (centre)
    180° pulse: 2500 µs

Environment variables (optional overrides):
    SERVO_GPIO_CHIP   — lgpio gpiochip index (default 0)
    SERVO_PAN_GPIO    — BCM pin for pan  (default 12)
    SERVO_TILT_GPIO   — BCM pin for tilt (default 13)
    SERVO_PAN_INVERT  — set "1" to reverse pan direction
    SERVO_TILT_INVERT — set "1" to reverse tilt direction
"""
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# ── Hardware constants ────────────────────────────────────────────────────────
_SERVO_FREQ_HZ = 50     # 50 Hz = standard servo PWM
_SERVO_MIN_US  = 500    # pulse width at 0°
_SERVO_MAX_US  = 2500   # pulse width at 180°

# ── Configuration from environment ────────────────────────────────────────────
_GPIO_CHIP   = int(os.environ.get("SERVO_GPIO_CHIP",   "0"))
PAN_GPIO     = int(os.environ.get("SERVO_PAN_GPIO",   "12"))
TILT_GPIO    = int(os.environ.get("SERVO_TILT_GPIO",  "13"))
PAN_INVERT   = os.environ.get("SERVO_PAN_INVERT",  "0").strip() == "1"
TILT_INVERT  = os.environ.get("SERVO_TILT_INVERT", "0").strip() == "1"

# ── Safe mechanical angle limits (degrees) ────────────────────────────────────
# Tune these to match your 3D-printed head geometry so servos never bind.
PAN_MIN_DEG  = 30
PAN_MAX_DEG  = 150
TILT_MIN_DEG = 50
TILT_MAX_DEG = 130

PAN_CENTER_DEG  = 90
TILT_CENTER_DEG = 90

# ── lgpio availability ────────────────────────────────────────────────────────
try:
    import lgpio as _lgpio  # type: ignore[import]
    _LGPIO_AVAILABLE = True
except ImportError:
    _lgpio = None           # type: ignore[assignment]
    _LGPIO_AVAILABLE = False


def _deg_to_us(deg: float) -> int:
    """Convert an angle in degrees [0, 180] to a pulse width in microseconds."""
    deg = max(0.0, min(180.0, deg))
    return int(_SERVO_MIN_US + (deg / 180.0) * (_SERVO_MAX_US - _SERVO_MIN_US))


class ServoController:
    """Thread-safe pan/tilt servo controller.

    Gracefully degrades to a stub when lgpio is unavailable (dev machines).
    """

    def __init__(
        self,
        pan_gpio:  int = PAN_GPIO,
        tilt_gpio: int = TILT_GPIO,
    ) -> None:
        self._pan_gpio  = pan_gpio
        self._tilt_gpio = tilt_gpio
        self._lock      = threading.Lock()
        self._handle    = None
        self._available = False
        self._pan_deg   = float(PAN_CENTER_DEG)
        self._tilt_deg  = float(TILT_CENTER_DEG)

        if not _LGPIO_AVAILABLE:
            logger.warning("[Servo] lgpio not available — running in stub mode")
            return

        try:
            self._handle = _lgpio.gpiochip_open(_GPIO_CHIP)
            _lgpio.gpio_claim_output(self._handle, pan_gpio)
            _lgpio.gpio_claim_output(self._handle, tilt_gpio)
            self._available = True
            logger.info(
                "[Servo] Ready: pan=GPIO%d, tilt=GPIO%d (chip %d)",
                pan_gpio, tilt_gpio, _GPIO_CHIP,
            )
            self.center()
        except Exception as exc:
            logger.error("[Servo] Init failed: %s", exc)

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def pan(self) -> float:
        """Current pan angle in degrees."""
        with self._lock:
            return self._pan_deg

    @property
    def tilt(self) -> float:
        """Current tilt angle in degrees."""
        with self._lock:
            return self._tilt_deg

    def set_pan(self, deg: float) -> None:
        """Move pan servo to *deg* degrees (clamped to safe limits)."""
        deg = max(PAN_MIN_DEG, min(PAN_MAX_DEG, float(deg)))
        with self._lock:
            self._pan_deg = deg  # store logical angle so pan property is correct
            phys = PAN_MIN_DEG + PAN_MAX_DEG - deg if PAN_INVERT else deg
            self._write(self._pan_gpio, phys)

    def set_tilt(self, deg: float) -> None:
        """Move tilt servo to *deg* degrees (clamped to safe limits)."""
        deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, float(deg)))
        with self._lock:
            self._tilt_deg = deg  # store logical angle so tilt property is correct
            phys = TILT_MIN_DEG + TILT_MAX_DEG - deg if TILT_INVERT else deg
            self._write(self._tilt_gpio, phys)

    def center(self) -> None:
        """Move both servos to their centre positions."""
        self.set_pan(PAN_CENTER_DEG)
        self.set_tilt(TILT_CENTER_DEG)

    def release(self) -> None:
        """Stop PWM on both servos so they hold position silently.

        The servo physically stays where it is (gear train holds it) but
        the control loop stops running, eliminating the jitter buzz.
        Call move methods to re-energise before the next movement.
        """
        if not self._available:
            return
        try:
            _lgpio.tx_servo(self._handle, self._pan_gpio,  0, _SERVO_FREQ_HZ)
            _lgpio.tx_servo(self._handle, self._tilt_gpio, 0, _SERVO_FREQ_HZ)
            logger.debug("[Servo] PWM released — servos silent")
        except Exception as exc:
            logger.error("[Servo] Release error: %s", exc)

    def shutdown(self) -> None:
        """Centre servos, wait for movement, stop PWM, release GPIO chip."""
        self.center()
        time.sleep(0.5)   # give servos time to reach centre
        self.release()
        if self._available and self._handle is not None:
            try:
                _lgpio.gpiochip_close(self._handle)
                logger.info("[Servo] Shutdown complete")
            except Exception as exc:
                logger.error("[Servo] Shutdown error: %s", exc)
            self._available = False

    # ── internal ───────────────────────────────────────────────────────────────

    def _write(self, gpio: int, deg: float) -> None:
        """Send PWM pulse to *gpio* for *deg* degrees. Must be called under lock."""
        if not self._available:
            logger.debug("[Servo stub] GPIO%d → %.1f°", gpio, deg)
            return
        try:
            _lgpio.tx_servo(self._handle, gpio, _deg_to_us(deg), _SERVO_FREQ_HZ)
        except Exception as exc:
            logger.error("[Servo] Write error GPIO%d: %s", gpio, exc)
