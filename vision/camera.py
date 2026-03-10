"""Camera for Tres-Robo: Raspberry Pi Camera Module 2 via picamera2.

Primary setup: Raspberry Pi + Camera Module 2 (CSI). Uses libcamera/picamera2.
OpenCV USB webcam is only a development fallback when picamera2 is not available
(e.g. running on Windows/Mac). On the Pi, install: sudo apt install -y python3-picamera2
and enable the camera (raspi-config → Interface Options → Camera).
"""
import os
import sys
import time
from contextlib import contextmanager
from types import TracebackType

import numpy as np


def _opencv_camera_index() -> int:
    """Camera index for OpenCV: 0 by default, overridable via CV2_CAMERA_INDEX."""
    try:
        return int(os.environ.get("CV2_CAMERA_INDEX", "0"))
    except ValueError:
        return 0


def _try_gstreamer_libcamera() -> "cv2.VideoCapture | None":
    """Pi Camera Module 2 via GStreamer+libcamera when picamera2 not installed. Returns None if not available."""
    import cv2  # type: ignore[import]

    # Only on Linux (Pi); needs gstreamer1.0-libcamera
    if sys.platform != "linux":
        return None
    # Try pipelines in order; libcamerasrc format varies by Pi OS version
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
    """Context manager: redirect stderr to devnull to silence OpenCV/V4L2/obsensor probe spam."""
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


def _find_working_camera() -> "tuple[cv2.VideoCapture, str]":
    """Kokeilee: GStreamer+libcamera (Pi), sitten OpenCV indeksit 0,1,2. Returns (cap, backend)."""
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
        "On Raspberry Pi with Pi Camera Module 2: sudo apt install -y python3-picamera2 (recommended), or "
        "sudo apt install gstreamer1.0-libcamera gstreamer1.0-plugins-good. "
        "Enable camera: raspi-config → Interface Options → Camera."
    )


class Camera:
    """Context manager that keeps the camera open for multiple captures.

    Usage:
        with Camera() as cam:
            frame = cam.capture()   # numpy BGR array (H, W, 3)
    """

    def __init__(self, warmup_seconds: float = 0.5) -> None:
        self._warmup = warmup_seconds
        self._backend: str = ""
        self._cam = None  # picamera2 handle
        self._cap = None  # cv2 VideoCapture handle

    def __enter__(self) -> "Camera":
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._close()

    # ------------------------------------------------------------------
    def capture(self) -> np.ndarray:
        """Return a single BGR frame as a numpy array (H, W, 3) uint8."""
        if self._backend == "picamera2":
            return self._capture_picamera2()
        return self._capture_opencv()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore[import]

            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (640, 480)}
            )
            cam.configure(config)
            cam.start()
            time.sleep(self._warmup)
            self._cam = cam
            self._backend = "picamera2"
            print("[Camera] Using Raspberry Pi Camera Module 2 (picamera2)")
        except ImportError:
            print(
                "[Camera] picamera2 not installed — Pi Camera Module 2 is the intended camera. "
                "On Raspberry Pi: sudo apt install -y python3-picamera2. "
                "Falling back to USB webcam (dev only).",
                file=sys.stderr,
            )
            self._open_opencv()
        except Exception as exc:
            print(
                f"[Camera] picamera2 failed ({exc}). "
                "Intended hardware: Raspberry Pi + Camera Module 2. Falling back to USB webcam (dev only).",
                file=sys.stderr,
            )
            self._open_opencv()

    def _open_opencv(self) -> None:
        cap, backend = _find_working_camera()
        time.sleep(self._warmup)
        # Discard first frames — many USB webcams return invalid/black frames until warmed up.
        for _ in range(5):
            cap.read()
        self._cap = cap
        self._backend = "opencv"  # both gstreamer and opencv use cap.read()
        if backend == "gstreamer":
            print("[Camera] Using GStreamer+libcamera (Pi Camera Module 2, picamera2 not installed)")
        else:
            idx = _opencv_camera_index()
            print(f"[Camera] Using OpenCV USB webcam (dev fallback; intended: Pi Camera Module 2, index={idx})")

    def _close(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_picamera2(self) -> np.ndarray:
        import cv2  # type: ignore[import]

        frame = self._cam.capture_array()
        # video_configuration returns XRGB or BGR depending on Pi OS version.
        # Normalise to BGR (3-channel) for OpenCV/JPEG encoding.
        if frame.ndim == 3 and frame.shape[2] == 4:
            # XBGR8888 on little-endian ARM: memory order is [R, G, B, X].
            # Drop X (channel 3) and swap R↔B to get BGR for OpenCV.
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # Assume RGB from libcamera → BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _capture_opencv(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame
        # First read often fails on USB cams; retry a few times.
        for attempt in range(4):
            time.sleep(0.1)
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
        raise RuntimeError("Failed to read frame from camera. Try another app closing the camera, or set CV2_CAMERA_INDEX=1 in .env.")
