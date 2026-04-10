"""Face-tracking pan/tilt servo control for Tres-Robo.

Runs as an independent background daemon thread — completely decoupled from
the cloud LLM conversation pipeline.

Behaviour
---------
ACTIVE (conversation mode)
    Opens the camera, captures frames at ~5 FPS, picks the largest
    (= closest) detected person bounding box, and moves the servo head
    proportionally to keep that person centred in frame.

INACTIVE (sleep mode)
    Releases the camera and moves both servos back to their centre
    positions.

Camera sharing
--------------
The 'see' tool in vision/scene.py also needs the camera.  Call
``face_tracker.camera_exclusive()`` as a context manager around any
code that must have exclusive camera access:

    with face_tracker.camera_exclusive():
        frame = capture_and_describe(...)

This will pause the tracking loop and wait until the camera is
physically released before yielding.

Tuning
------
Adjust the constants below for your specific head geometry and
mounting orientation.  If the head moves in the wrong direction,
set SERVO_PAN_INVERT=1 or SERVO_TILT_INVERT=1 in your .env file.
"""
import logging
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── Tracking parameters ───────────────────────────────────────────────────────

# How often to run one tracking step (seconds).  Lower = more responsive
# but more CPU and camera load.  0.20 s ≈ 5 FPS.
_INTERVAL_S = 0.20

# Proportional gain: degrees of servo movement per pixel of tracking error.
# Increase for faster response; decrease if head oscillates.
_P_GAIN = 0.04

# Maximum servo movement per step (degrees).  Caps sudden jumps.
_MAX_STEP_DEG = 8.0

# Pixel deadband around frame centre: errors smaller than this are ignored.
# Prevents micro-jitter when the person is already well-centred.
_DEADBAND_PX = 25

# Expected frame resolution (must match Camera output, default 640×480).
_FRAME_W = 640
_FRAME_H = 480

# How far down the person bounding box the face centre is estimated to be.
# 0.20 = upper 20 % of the box ≈ head region.
_FACE_Y_FRACTION = 0.20

# Camera warmup when (re-)opening.  Shorter than the default 0.5 s since
# we only need object detection, not a perfect exposure.
_CAMERA_WARMUP_S = 0.30

# COCO class index for "person"
_PERSON_CLASS = 0


class FaceTracker:
    """Background thread that tracks the closest detected person.

    Args:
        servo: A ``hardware.servo.ServoController`` instance.
    """

    def __init__(self, servo) -> None:
        self._servo     = servo

        # Conversation state: set = active tracking, clear = sleep/centre
        self._active    = threading.Event()

        # Pause handshake for camera_exclusive():
        #   _pause_req  — external code requests pause
        #   _paused     — tracker confirms camera is released
        self._pause_req = threading.Event()
        self._paused    = threading.Event()

        self._stop      = threading.Event()
        self._thread    = threading.Thread(
            target=self._loop,
            name="face-tracker",
            daemon=True,
        )

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background tracking thread."""
        self._thread.start()
        logger.info("[FaceTracker] Thread started")

    def set_active(self, active: bool) -> None:
        """Switch between conversation (tracking) and sleep (centre) mode."""
        if active:
            self._active.set()
            logger.info("[FaceTracker] Tracking ON")
        else:
            self._active.clear()
            logger.info("[FaceTracker] Tracking OFF → centering")

    def stop(self) -> None:
        """Signal the thread to exit and wait for it."""
        self._stop.set()
        self._thread.join(timeout=4.0)

    @contextmanager
    def camera_exclusive(self):
        """Context manager granting exclusive camera access to the caller.

        Pauses face tracking, waits until the camera is released, then
        yields.  Tracking resumes automatically when the context exits.

        Usage::

            with face_tracker.camera_exclusive():
                with Camera() as cam:
                    frame = cam.capture()
        """
        # If not currently tracking the camera is already free.
        if not self._active.is_set():
            yield
            return

        self._pause_req.set()
        if not self._paused.wait(timeout=3.0):
            logger.warning("[FaceTracker] camera_exclusive: timeout waiting for release")
        try:
            yield
        finally:
            self._pause_req.clear()
            self._paused.clear()

    # ── main loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():

            # --- Pause gate (checked before trying to open the camera) --------
            if self._pause_req.is_set():
                self._paused.set()
                while self._pause_req.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                self._paused.clear()
                continue

            # --- Sleep mode: centre and wait ----------------------------------
            if not self._active.is_set():
                self._servo.center()
                self._active.wait(timeout=0.5)
                continue

            # --- Active mode: open camera and track ---------------------------
            try:
                from vision.camera import Camera  # local import avoids circular deps

                with Camera(warmup_seconds=_CAMERA_WARMUP_S) as cam:
                    if cam.has_ai:
                        print("[FaceTracker] Camera opened (IMX500 AI — person detection active)")
                    else:
                        print(
                            f"[FaceTracker] Camera opened (backend has no on-device detection — "
                            "person tracking will not work; requires IMX500 AI Camera)"
                        )

                    while self._active.is_set() and not self._stop.is_set():

                        # Pause gate inside the camera-open inner loop
                        if self._pause_req.is_set():
                            logger.debug("[FaceTracker] Pausing (camera_exclusive requested)")
                            self._paused.set()
                            while self._pause_req.is_set() and not self._stop.is_set():
                                time.sleep(0.05)
                            self._paused.clear()
                            break  # exit inner loop → camera closes → outer loop re-opens

                        self._track_once(cam)
                        time.sleep(_INTERVAL_S)

                    logger.debug("[FaceTracker] Camera releasing")

            except Exception as exc:
                logger.warning("[FaceTracker] Camera error: %s — retrying in 1.5 s", exc)
                time.sleep(1.5)

    # ── tracking step ──────────────────────────────────────────────────────────

    def _track_once(self, cam) -> None:
        """Capture one frame and nudge the servos toward the closest person."""
        try:
            # Lower threshold slightly for debug visibility — default 0.5 can miss
            # close-up faces/heads that are only partially in frame.
            _, detections = cam.capture_with_detections(min_score=0.4)
        except Exception as exc:
            print(f"[FaceTracker] Capture failed: {exc}")
            return

        if not detections:
            print("[FaceTracker] No detections at all (score ≥ 0.4)")
            return

        # Log every detection so we can see what the model actually found
        for d in detections:
            print(
                f"[FaceTracker] Detection: class={d['class']} score={d.get('score', 0):.2f} "
                f"box={d['box']}"
            )

        # Filter to person class only
        persons = [d for d in detections if d.get("class") == _PERSON_CLASS]
        if not persons:
            print(f"[FaceTracker] No person (class 0) — got classes: {[d['class'] for d in detections]}")
            return

        if len(persons) > 1:
            print(f"[FaceTracker] {len(persons)} persons detected — tracking closest (largest box)")

        # Pick the largest bounding box (largest area ≈ closest person)
        best = max(persons, key=lambda d: _box_area(d["box"]))
        x1, y1, x2, y2 = best["box"]
        score = best.get("score", 0.0)

        # The IMX500 runs inference on the original (pre-rotation) frame.
        # camera.py rotates the frame 180° in software, so the detection boxes
        # are in the flipped coordinate space and must be transformed to match.
        x1, y1, x2, y2 = _FRAME_W - x2, _FRAME_H - y2, _FRAME_W - x1, _FRAME_H - y1
        area = int(_box_area((x1, y1, x2, y2)))

        # Estimate face centre: upper portion of the person bounding box
        face_cx = (x1 + x2) / 2.0
        face_cy = y1 + (y2 - y1) * _FACE_Y_FRACTION

        print(
            f"[FaceTracker] Person @ ({face_cx:.0f}, {face_cy:.0f})  "
            f"box={x1},{y1}-{x2},{y2}  area={area}  score={score:.2f}"
        )

        # Tracking error from frame centre
        # Positive err_x → face is to the RIGHT  → pan right (increase pan angle)
        # Positive err_y → face is BELOW centre  → tilt down (increase tilt angle)
        err_x = face_cx - _FRAME_W / 2.0
        err_y = face_cy - _FRAME_H / 2.0

        # Apply deadband
        if abs(err_x) < _DEADBAND_PX:
            err_x = 0.0
        if abs(err_y) < _DEADBAND_PX:
            err_y = 0.0

        if err_x == 0.0 and err_y == 0.0:
            print(f"[FaceTracker] Centred (pan={self._servo.pan:.1f}°, tilt={self._servo.tilt:.1f}°) — no move")
            return  # already centred — nothing to do

        # Proportional control, step-clamped for smooth motion
        delta_pan  = _clamp(err_x * _P_GAIN, -_MAX_STEP_DEG, _MAX_STEP_DEG)
        delta_tilt = _clamp(err_y * _P_GAIN, -_MAX_STEP_DEG, _MAX_STEP_DEG)

        pan_before  = self._servo.pan
        tilt_before = self._servo.tilt
        self._servo.set_pan(pan_before   + delta_pan)
        self._servo.set_tilt(tilt_before + delta_tilt)

        print(
            f"[FaceTracker] err=({err_x:+.0f}px, {err_y:+.0f}px)  "
            f"pan {pan_before:.1f}°→{self._servo.pan:.1f}° ({delta_pan:+.2f}°)  "
            f"tilt {tilt_before:.1f}°→{self._servo.tilt:.1f}° ({delta_tilt:+.2f}°)"
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
