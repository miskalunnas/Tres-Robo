"""Face detection and recognition using the face_recognition library (dlib).

Typical usage:
    locations = detect(frame)
    encodings = encode(frame, locations)
    person = db.identify(encodings[0])
"""
import numpy as np

# face_recognition is a heavy install (requires dlib + cmake).
# We import lazily and give a clear error if it's missing.
try:
    import face_recognition as fr  # type: ignore[import]
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False


def _require_fr() -> None:
    if not _FR_AVAILABLE:
        raise RuntimeError(
            "face_recognition is not installed.\n"
            "On Pi: sudo apt install cmake libopenblas-dev liblapack-dev\n"
            "       pip install face_recognition"
        )


def detect(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return a list of face bounding boxes as (top, right, bottom, left) tuples.

    Uses HOG model (fast, CPU-friendly). Pass model='cnn' for better accuracy
    at the cost of ~3x more compute.
    """
    _require_fr()
    # np.ascontiguousarray ensures dlib gets a C-contiguous array ([::-1] gives a view)
    frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR → RGB
    return fr.face_locations(frame_rgb, model="hog")


def encode(
    frame_bgr: np.ndarray,
    locations: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """Return 128-d face embeddings for the given face locations."""
    _require_fr()
    frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return fr.face_encodings(frame_rgb, known_face_locations=locations)


def detect_and_encode(frame_bgr: np.ndarray) -> list[np.ndarray]:
    """Convenience: detect all faces in a frame and return their encodings."""
    locations = detect(frame_bgr)
    if not locations:
        return []
    return encode(frame_bgr, locations)
