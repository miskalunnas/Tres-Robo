"""Background helper that binds a recognized person to the active session."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
import sys


class IdentityWatcher:
    """Periodically sample the camera and bind a recognized person."""

    def __init__(
        self,
        on_person: Callable[[str | None], None],
        *,
        interval_seconds: float = 2.5,
    ) -> None:
        self._on_person = on_person
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            from vision.camera import Camera
            from vision.face_db import FaceDB
            from vision.face_id import detect_and_encode
        except Exception as exc:
            print(f"[Vision] Live identity unavailable: {exc}", file=sys.stderr)
            return

        face_db = FaceDB()
        last_person_id: str | None = None

        try:
            with Camera() as camera:
                while not self._stop_event.is_set():
                    frame = camera.capture()
                    encodings = detect_and_encode(frame)
                    identified_person_id = None
                    for encoding in encodings:
                        person = face_db.identify(encoding)
                        if person is not None:
                            identified_person_id = person.id
                            break

                    if identified_person_id and identified_person_id != last_person_id:
                        self._on_person(identified_person_id)
                        last_person_id = identified_person_id

                    self._stop_event.wait(self._interval_seconds)
        except Exception as exc:
            print(f"[Vision] Live identity stopped: {exc}", file=sys.stderr)
