"""Camera abstraction — tries picamera2 (Pi Camera Module) first, falls back to OpenCV."""
import sys
import time
from types import TracebackType

import numpy as np


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
            print("[Camera] Using picamera2 (Pi Camera Module)")
        except ImportError:
            print("[Camera] picamera2 not installed — falling back to OpenCV USB webcam.", file=sys.stderr)
            self._open_opencv()
        except Exception as exc:
            print(f"[Camera] picamera2 failed ({exc}) — falling back to OpenCV USB webcam.", file=sys.stderr)
            self._open_opencv()

    def _open_opencv(self) -> None:
        import cv2  # type: ignore[import]

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "No camera found. Install picamera2 or connect a USB webcam."
            )
        time.sleep(self._warmup)
        self._cap = cap
        self._backend = "opencv"
        print("[Camera] Using OpenCV (USB/webcam fallback)")

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
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame
