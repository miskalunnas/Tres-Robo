"""Face-tracking pan/tilt servo control for Tres-Robo.

Runs as an independent background daemon thread — completely decoupled from
the cloud LLM conversation pipeline.

Behaviour
---------
ACTIVE (conversation mode)
    Opens the camera, captures frames at ~5 FPS, detects faces, picks the
    largest (= closest) one, and moves the servo head proportionally to keep
    it centred in frame.

    Detection backend (chosen automatically):
      IMX500 AI Camera  — on-device EfficientDet person detection (preferred)
      Standard camera   — OpenCV Haar cascade face detector (fallback)

INACTIVE (sleep mode)
    Releases the camera and moves both servos back to their centre positions.

Camera sharing
--------------
The 'see' tool in vision/scene.py also needs the camera.  Call
``face_tracker.camera_exclusive()`` as a context manager around any
code that must have exclusive camera access:

    with face_tracker.camera_exclusive():
        frame = capture_and_describe(...)

This will pause the tracking loop and wait until the camera is physically
released before yielding.

Tuning
------
Adjust the constants below for your specific head geometry and mounting
orientation.  If the head moves in the wrong direction, set
SERVO_PAN_INVERT=1 or SERVO_TILT_INVERT=1 in your .env file.
"""
import logging
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── Tracking parameters ───────────────────────────────────────────────────────

# How often to run one tracking step (seconds).  0.20 s ≈ 5 FPS.
_INTERVAL_S = 0.20

# Proportional gain: degrees of servo movement per pixel of tracking error.
_P_GAIN = 0.04

# Maximum servo movement per step (degrees).
_MAX_STEP_DEG = 8.0

# Pixel deadband — errors smaller than this are ignored to prevent jitter.
_DEADBAND_PX = 25

# Expected frame resolution (must match Camera output, default 640×480).
_FRAME_W = 640
_FRAME_H = 480

# Camera warmup when (re-)opening.
_CAMERA_WARMUP_S = 0.30

# COCO class index for "person" (IMX500 backend)
_PERSON_CLASS = 0


# ── OpenCV Haar cascade face detector (software fallback) ─────────────────────

def _load_haar_cascade():
    """Return an OpenCV face CascadeClassifier, or None if unavailable."""
    try:
        import cv2
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None


def _haar_detect_faces(frame, cascade) -> list[dict]:
    """Run Haar cascade on *frame* and return detections in the same format
    as Camera.capture_with_detections() — list of {"box", "score", "class"}."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    detections = []
    if len(faces):
        for (x, y, w, h) in faces:
            detections.append({
                "box":   (int(x), int(y), int(x + w), int(y + h)),
                "score": 1.0,   # Haar doesn't produce a confidence score
                "class": _PERSON_CLASS,
            })
    return detections


class FaceTracker:
    """Background thread that tracks the closest detected face.

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
            print("[FaceTracker] Tracking ON")
        else:
            self._active.clear()
            print("[FaceTracker] Tracking OFF → centering")

    def stop(self) -> None:
        """Signal the thread to exit and wait for it."""
        self._stop.set()
        self._thread.join(timeout=4.0)

    @contextmanager
    def camera_exclusive(self):
        """Context manager granting exclusive camera access to the caller.

        Pauses face tracking, waits until the camera is released, then
        yields.  Tracking resumes automatically when the context exits.
        """
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

            # --- Pause gate ---------------------------------------------------
            if self._pause_req.is_set():
                self._paused.set()
                while self._pause_req.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                self._paused.clear()
                continue

            # --- Sleep mode ---------------------------------------------------
            if not self._active.is_set():
                self._servo.center()
                time.sleep(0.6)      # let servos physically reach centre
                self._servo.release()  # cut PWM — stops jitter buzz
                self._active.wait()  # block indefinitely until conversation starts
                continue

            # --- Active mode: open camera and track ---------------------------
            try:
                from vision.camera import Camera

                with Camera(warmup_seconds=_CAMERA_WARMUP_S) as cam:
                    if cam.has_ai:
                        print("[FaceTracker] Camera opened — using IMX500 on-device detection")
                        haar = None
                    else:
                        haar = _load_haar_cascade()
                        if haar:
                            print("[FaceTracker] Camera opened — using OpenCV Haar cascade (software fallback)")
                        else:
                            print("[FaceTracker] Camera opened — no detector available (no IMX500, no OpenCV)")

                    while self._active.is_set() and not self._stop.is_set():
                        if self._pause_req.is_set():
                            break  # exit inner loop → with-block closes camera below
                        self._track_once(cam, haar)
                        time.sleep(_INTERVAL_S)

                # Camera is now physically closed — safe to signal the caller.
                if self._pause_req.is_set():
                    print("[FaceTracker] Camera released — handing over to see tool")
                    self._paused.set()
                    while self._pause_req.is_set() and not self._stop.is_set():
                        time.sleep(0.05)
                    self._paused.clear()
                    print("[FaceTracker] Resuming tracking")

            except Exception as exc:
                logger.warning("[FaceTracker] Camera error: %s — retrying in 1.5 s", exc)
                time.sleep(1.5)

    # ── tracking step ──────────────────────────────────────────────────────────

    def _track_once(self, cam, haar) -> None:
        """Capture one frame, detect faces, and nudge the servos."""
        try:
            frame, detections = cam.capture_with_detections(min_score=0.4)
        except Exception as exc:
            print(f"[FaceTracker] Capture failed: {exc}")
            return

        # IMX500 returned nothing — try Haar fallback on the same frame
        if not detections and haar is not None:
            detections = _haar_detect_faces(frame, haar)

        if not detections:
            print("[FaceTracker] No face detected")
            return

        # Filter / label
        persons = [d for d in detections if d.get("class") == _PERSON_CLASS]
        if not persons:
            print(f"[FaceTracker] No person (class 0) — classes seen: {[d['class'] for d in detections]}")
            return

        if len(persons) > 1:
            print(f"[FaceTracker] {len(persons)} faces — tracking closest (largest box)")

        # Largest box ≈ closest face
        best = max(persons, key=lambda d: _box_area(d["box"]))
        x1, y1, x2, y2 = best["box"]
        score = best.get("score", 1.0)

        # IMX500 inference runs on the pre-rotation frame; camera.py rotates
        # 180° in software, so we must mirror the box coordinates to match.
        # Haar runs on the already-rotated frame so no transform is needed.
        if cam.has_ai:
            x1, y1, x2, y2 = _FRAME_W - x2, _FRAME_H - y2, _FRAME_W - x1, _FRAME_H - y1

        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0  # face box centre (Haar gives the face directly)

        area = int(_box_area((x1, y1, x2, y2)))
        print(
            f"[FaceTracker] Face @ ({face_cx:.0f}, {face_cy:.0f})  "
            f"box={x1},{y1}-{x2},{y2}  area={area}  score={score:.2f}"
        )

        err_x = face_cx - _FRAME_W / 2.0
        err_y = face_cy - _FRAME_H / 2.0

        if abs(err_x) < _DEADBAND_PX:
            err_x = 0.0
        if abs(err_y) < _DEADBAND_PX:
            err_y = 0.0

        if err_x == 0.0 and err_y == 0.0:
            print(f"[FaceTracker] Centred (pan={self._servo.pan:.1f}°, tilt={self._servo.tilt:.1f}°)")
            return

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
