"""Servo test script — moves each servo slowly from limit to limit.

Run from the project root:
    python test_servos.py

Press Ctrl+C at any time to stop and centre both servos.
"""
import time
import sys

from hardware.servo import (
    ServoController,
    PAN_MIN_DEG, PAN_MAX_DEG, PAN_CENTER_DEG,
    TILT_MIN_DEG, TILT_MAX_DEG, TILT_CENTER_DEG,
)

STEP_DEG  = 1.0   # degrees per step — smaller = smoother/slower
STEP_DELAY = 0.03  # seconds between steps (~33 steps/sec)


def sweep(label: str, set_fn, min_deg: float, max_deg: float) -> None:
    """Sweep from min to max and back once, printing position."""
    print(f"\n[Test] {label}: {min_deg:.0f}° → {max_deg:.0f}°")
    deg = min_deg
    while deg <= max_deg:
        set_fn(deg)
        print(f"  {label}: {deg:.1f}°", end="\r")
        deg += STEP_DEG
        time.sleep(STEP_DELAY)

    print(f"\n[Test] {label}: {max_deg:.0f}° → {min_deg:.0f}°")
    deg = max_deg
    while deg >= min_deg:
        set_fn(deg)
        print(f"  {label}: {deg:.1f}°", end="\r")
        deg -= STEP_DEG
        time.sleep(STEP_DELAY)
    print()


def main() -> None:
    print("[Test] Initialising servos...")
    servo = ServoController()

    if not servo._available:
        print("[Test] WARNING: lgpio not available — running in stub mode (no physical movement)")

    try:
        print(f"[Test] Centring both servos (pan={PAN_CENTER_DEG}°, tilt={TILT_CENTER_DEG}°)")
        servo.center()
        time.sleep(1.0)

        # --- Pan servo ---
        print("\n[Test] ── PAN servo (tilt held at centre) ──")
        sweep("Pan", servo.set_pan, PAN_MIN_DEG, PAN_MAX_DEG)
        servo.center()
        time.sleep(0.5)

        # --- Tilt servo ---
        print("\n[Test] ── TILT servo (pan held at centre) ──")
        sweep("Tilt", servo.set_tilt, TILT_MIN_DEG, TILT_MAX_DEG)
        servo.center()
        time.sleep(0.5)

        print("\n[Test] Done. Both servos centred.")

    except KeyboardInterrupt:
        print("\n[Test] Interrupted — centring and shutting down.")
    finally:
        servo.shutdown()


if __name__ == "__main__":
    main()
