"""FaceManager: in-memory face recognition with persistent enrollment.

Architecture
------------
- Enrollment writes 128-d face_recognition encodings to robot.db via FaceDB.
- At startup (or after enrollment) the full encoding table is loaded into
  self._known — a flat list of (name, encoding) pairs.  All live recognition
  runs entirely in RAM; the DB is only hit on load/register.
- Multiple encodings per person are supported automatically.

Quick-start
-----------
Register from a still image (e.g. captured with rpicam-still):
    from vision.identity_manager import FaceManager
    FaceManager.get().register_face("Miro", "/tmp/miro.jpg")

Or use the bundled enrollment script:
    python vision/enroll.py --name "Miro"    # captures 5 frames from the camera
"""

import sys
from pathlib import Path

import numpy as np

MATCH_THRESHOLD = 0.55  # Euclidean distance in 128-d space; ≤ 0.55 = match


class FaceManager:
    """Singleton orchestrator for face detection, encoding, and recognition."""

    _instance: "FaceManager | None" = None

    @classmethod
    def get(cls) -> "FaceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # In-memory cache: flat list of (display_name, encoding_array).
        # Multiple entries per name = ensemble matching.
        self._known: list[tuple[str, np.ndarray]] = []
        self._load_from_db()

    # ------------------------------------------------------------------
    # Public API

    def register_face(self, name: str, image_path: str | Path) -> bool:
        """Detect a face in *image_path*, encode it, and persist to DB.

        Adds to the existing person if *name* already exists so you can build
        up an ensemble by calling this with several photos of the same person.
        Returns True if a face was found and saved.
        """
        import cv2
        from vision.face_db import FaceDB
        from vision import face_id

        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            print(f"[FaceManager] Could not load image: {image_path}", file=sys.stderr)
            return False

        encodings = face_id.detect_and_encode(frame_bgr)
        if not encodings:
            print(f"[FaceManager] No face detected in {image_path}", file=sys.stderr)
            return False

        if len(encodings) > 1:
            print(f"[FaceManager] Multiple faces found — using the largest/first one")

        db = FaceDB()
        # Add to existing person or create new one.
        existing = [p for p in db.list_persons() if p.name.lower() == name.lower()]
        if existing:
            from memory import MemoryStore
            MemoryStore().add_face_embedding(
                existing[0].id, encodings[0], source="register_face", threshold=MATCH_THRESHOLD
            )
            print(f"[FaceManager] Added encoding to existing person '{name}' ({existing[0].id})")
        else:
            db.enrol(name, [encodings[0]])

        self.reload()
        return True

    def load_known_faces(self) -> int:
        """Reload from DB and return the number of encodings loaded."""
        self._load_from_db()
        return len(self._known)

    def reload(self) -> None:
        """Alias for load_known_faces() — call after new enrollments."""
        self._load_from_db()

    def recognize_faces(self, frame_bgr: np.ndarray) -> list[str]:
        """Detect and identify all faces in *frame_bgr*.

        Returns a deduplicated list of recognised display_names in the order
        they were found.  Returns [] if face_recognition is not installed or
        no known faces are enrolled.
        """
        if not self._known:
            return []

        try:
            import face_recognition as fr
        except ImportError:
            return []

        from vision import face_id
        try:
            encodings = face_id.detect_and_encode(frame_bgr)
        except Exception as exc:
            print(f"[FaceManager] Detection error: {exc}", file=sys.stderr)
            return []

        if not encodings:
            return []

        known_encs = np.array([e for _, e in self._known])
        known_names = [n for n, _ in self._known]

        recognized: list[str] = []
        seen: set[str] = set()

        for enc in encodings:
            distances = fr.face_distance(known_encs, enc)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= MATCH_THRESHOLD:
                name = known_names[best_idx]
                if name not in seen:
                    recognized.append(name)
                    seen.add(name)

        return recognized

    def list_enrolled(self) -> list[str]:
        """Return the names of all enrolled persons."""
        return sorted({name for name, _ in self._known})

    # ------------------------------------------------------------------
    # Internal

    def _load_from_db(self) -> None:
        from vision.face_db import FaceDB
        db = FaceDB()
        self._known = [
            (person.name, enc)
            for person in db.list_persons()
            for enc in person.embeddings
        ]
        n_persons = len({n for n, _ in self._known})
        print(
            f"[FaceManager] Loaded {len(self._known)} encoding(s) "
            f"for {n_persons} person(s)"
        )
