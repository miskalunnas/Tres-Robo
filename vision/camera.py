"""Camera for Tres-Robo: Raspberry Pi AI Camera (IMX500) via picamera2.

Primary setup: Raspberry Pi + AI Camera Module (IMX500, CSI). Uses libcamera/picamera2.
Provides standard frame capture plus on-device neural network inference for person/object
detection via capture_with_detections() and person_detections().

Falls back to standard picamera2 (Camera Module 2 etc.) if IMX500 is not detected,
and to OpenCV USB webcam for development on non-Pi machines.

Install on Pi:
    sudo apt install -y python3-picamera2 imx500-all
    raspi-config → Interface Options → Camera (enable)
"""
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

import numpy as np

# Global lock — picamera2 only allows one open instance at a time.
# All Camera context-manager users automatically serialise through this.
_camera_lock = threading.Lock()

# ── Constants ──────────────────────────────────────────────────────────────────

# Default on-device model: EfficientDet Lite0 (COCO 80 classes, includes "person")
_AI_MODEL = "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk"

# COCO class index for person
_PERSON_CLASS = 0


# ── OpenCV helpers (dev fallback) ──────────────────────────────────────────────

def _opencv_camera_index() -> int:
    try:
        return int(os.environ.get("CV2_CAMERA_INDEX", "0"))
    except ValueError:
        return 0


def _try_gstreamer_libcamera() -> "cv2.VideoCapture | None":
    """Pi Camera via GStreamer+libcamera when picamera2 is not installed."""
    import cv2  # type: ignore[import]

    if sys.platform != "linux":
        return None
    pipelines = [
        "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink",
        "libcamerasrc ! video/x-raw,width=640,height=480,framerate=15/1 ! videoconvert ! appsink",
        "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink",
    ]
    for pipeline in pipelines:
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        except Exception:
            pass
    return None


@contextmanager
def _suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull)


def _find_working_opencv_camera() -> "tuple[cv2.VideoCapture, str]":
    import cv2  # type: ignore[import]

    cap = _try_gstreamer_libcamera()
    if cap is not None:
        return cap, "gstreamer"

    preferred = _opencv_camera_index()
    indices = [preferred] + [i for i in (0, 1, 2) if i != preferred]
    for idx in indices:
        with _suppress_stderr():
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, "opencv"
                cap.release()
    raise RuntimeError(
        f"No working camera at indices {indices}. "
        "On Raspberry Pi: sudo apt install -y python3-picamera2 imx500-all. "
        "Enable camera: raspi-config → Interface Options → Camera."
    )


# ── Camera ─────────────────────────────────────────────────────────────────────

class Camera:
    """Context manager that keeps the camera open for multiple captures.

    Usage:
        with Camera() as cam:
            frame = cam.capture()                        # BGR numpy array (H, W, 3)
            frame, detections = cam.capture_with_detections()  # + on-device inference
            frame, persons    = cam.person_detections()        # filtered to people only

    Detection dict keys:
        "box":   (x1, y1, x2, y2) in pixels
        "score": float confidence 0–1
        "class": int COCO class index (0 = person)
    """

    def __init__(self, warmup_seconds: float = 0.5, model_path: str = _AI_MODEL) -> None:
        self._warmup = warmup_seconds
        self._model_path = model_path
        self._backend: str = ""
        self._cam = None       # Picamera2 instance
        self._cap = None       # cv2.VideoCapture instance (dev fallback)
        self._imx500 = None    # IMX500 instance (AI Camera only)

    def __enter__(self) -> "Camera":
        _camera_lock.acquire()
        try:
            self._open()
        except Exception:
            _camera_lock.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            self._close()
        finally:
            _camera_lock.release()

    # ── public API ─────────────────────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """Return a single BGR frame as a numpy array (H, W, 3) uint8."""
        import cv2  # type: ignore[import]

        if self._backend in ("imx500", "picamera2"):
            frame = self._capture_picamera2()
        else:
            frame = self._capture_opencv()
        # Camera is mounted upside-down — flip here once so all callers get the right orientation.
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def capture_with_detections(
        self, min_score: float = 0.5
    ) -> "tuple[np.ndarray, list[dict]]":
        """Capture a frame and return on-device inference detections (AI Camera only).

        On non-IMX500 backends detections is always [].

        Returns:
            (frame, detections)
            frame:      BGR numpy array (H, W, 3)
            detections: list of {"box": (x1,y1,x2,y2), "score": float, "class": int}
        """
        frame = self.capture()
        if self._backend != "imx500" or self._imx500 is None:
            return frame, []
        try:
            metadata = self._cam.capture_metadata()
            np_outputs = self._imx500.get_outputs(metadata, np_outputs=True)
            if np_outputs is None:
                return frame, []

            # EfficientDet lite0 pp outputs: boxes [N,4], scores [N], classes [N]
            # boxes are normalized [y_min, x_min, y_max, x_max]
            boxes, scores, classes = np_outputs[0], np_outputs[1], np_outputs[2]
            h, w = frame.shape[:2]
            detections = []
            for box, score, cls in zip(boxes, scores, classes):
                if float(score) < min_score:
                    continue
                y1, x1, y2, x2 = box
                detections.append({
                    "box":   (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)),
                    "score": float(score),
                    "class": int(cls),
                })
            return frame, detections
        except Exception as exc:
            print(f"[Camera] Detection read failed: {exc}", file=sys.stderr)
            return frame, []

    def person_detections(
        self, min_score: float = 0.5
    ) -> "tuple[np.ndarray, list[dict]]":
        """Like capture_with_detections() but filtered to person class (COCO class 0) only."""
        frame, dets = self.capture_with_detections(min_score)
        persons = [d for d in dets if d["class"] == _PERSON_CLASS]
        return frame, persons

    @property
    def has_ai(self) -> bool:
        """True when the AI Camera (IMX500) is active and on-device inference is available."""
        return self._backend == "imx500"

    # ── open / close ───────────────────────────────────────────────────────────

    def _open(self) -> None:
        # 1. Try AI Camera (IMX500)
        if Path(self._model_path).exists():
            try:
                from picamera2.devices.imx500 import IMX500  # type: ignore[import]
                from picamera2 import Picamera2              # type: ignore[import]

                imx500 = IMX500(self._model_path)
                cam = Picamera2(imx500.camera_num)
                config = cam.create_video_configuration(main={"size": (640, 480)})
                cam.configure(config)
                cam.start()
                time.sleep(self._warmup)
                self._imx500 = imx500
                self._cam = cam
                self._backend = "imx500"
                print(
                    f"[Camera] Using Raspberry Pi AI Camera (IMX500) — "
                    f"model: {Path(self._model_path).name}"
                )
                return
            except ImportError:
                pass  # picamera2.devices.imx500 not installed → try standard picamera2
            except Exception as exc:
                print(
                    f"[Camera] IMX500 init failed ({exc}), trying standard picamera2.",
                    file=sys.stderr,
                )

        # 2. Try standard picamera2 (Camera Module 2 or other CSI camera)
        try:
            from picamera2 import Picamera2  # type: ignore[import]

            cam = Picamera2()
            config = cam.create_video_configuration(main={"size": (640, 480)})
            cam.configure(config)
            cam.start()
            time.sleep(self._warmup)
            self._cam = cam
            self._backend = "picamera2"
            print("[Camera] Using Raspberry Pi Camera (picamera2, no on-device inference)")
            return
        except ImportError:
            print(
                "[Camera] picamera2 not installed. "
                "On Raspberry Pi: sudo apt install -y python3-picamera2 imx500-all. "
                "Falling back to USB webcam (dev only).",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[Camera] picamera2 failed ({exc}). Falling back to USB webcam (dev only).",
                file=sys.stderr,
            )

        # 3. Dev fallback: OpenCV USB webcam
        self._open_opencv()

    def _open_opencv(self) -> None:
        cap, backend = _find_working_opencv_camera()
        time.sleep(self._warmup)
        for _ in range(5):
            cap.read()
        self._cap = cap
        self._backend = "opencv"
        if backend == "gstreamer":
            print("[Camera] Using GStreamer+libcamera (picamera2 not installed)")
        else:
            idx = _opencv_camera_index()
            print(f"[Camera] Using OpenCV USB webcam (dev fallback, index={idx})")

    def _close(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._imx500 = None

    # ── frame capture internals ────────────────────────────────────────────────

    def _capture_picamera2(self) -> np.ndarray:
        import cv2  # type: ignore[import]

        frame = self._cam.capture_array()
        if frame.ndim == 3 and frame.shape[2] == 4:
            # XBGR8888 on little-endian ARM — drop X, swap R↔B
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _capture_opencv(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame
        for _ in range(4):
            time.sleep(0.1)
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
        raise RuntimeError(
            "Failed to read frame from camera. "
            "Check that nothing else is using the camera, or set CV2_CAMERA_INDEX=1 in .env."
        )
