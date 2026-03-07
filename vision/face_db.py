"""Persistent face database — stores name + face embeddings as JSON.

Database lives at data/faces.json (gitignored).
Only embeddings (128-d float vectors) are stored — no raw images.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent.parent / "data" / "faces.json"
SIMILARITY_THRESHOLD = 0.55  # Euclidean distance; lower = stricter match


@dataclass
class Person:
    id: str
    name: str
    enrolled_at: str
    embeddings: list[np.ndarray] = field(default_factory=list)


class FaceDB:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._persons: dict[str, Person] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API

    def enrol(self, name: str, embeddings: list[np.ndarray]) -> Person:
        """Add a new person with one or more face embeddings."""
        person_id = str(uuid.uuid4())[:8]
        person = Person(
            id=person_id,
            name=name,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
            embeddings=embeddings,
        )
        self._persons[person_id] = person
        self._save()
        print(f"[FaceDB] Enrolled '{name}' (id={person_id}, {len(embeddings)} embedding(s))")
        return person

    def identify(self, embedding: np.ndarray) -> Person | None:
        """Return the closest known person, or None if no match within threshold."""
        best_person: Person | None = None
        best_dist = float("inf")

        for person in self._persons.values():
            for known_emb in person.embeddings:
                dist = float(np.linalg.norm(embedding - known_emb))
                if dist < best_dist:
                    best_dist = dist
                    best_person = person

        if best_dist <= SIMILARITY_THRESHOLD:
            return best_person
        return None

    def list_persons(self) -> list[Person]:
        return list(self._persons.values())

    def remove(self, person_id: str) -> bool:
        if person_id in self._persons:
            name = self._persons.pop(person_id).name
            self._save()
            print(f"[FaceDB] Removed '{name}' (id={person_id})")
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for pid, entry in data.items():
                self._persons[pid] = Person(
                    id=pid,
                    name=entry["name"],
                    enrolled_at=entry["enrolled_at"],
                    embeddings=[np.array(e) for e in entry["embeddings"]],
                )
            print(f"[FaceDB] Loaded {len(self._persons)} person(s) from {self._path}")
        except Exception as exc:
            print(f"[FaceDB] Warning: could not load database: {exc}")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            pid: {
                "name": p.name,
                "enrolled_at": p.enrolled_at,
                "embeddings": [e.tolist() for e in p.embeddings],
            }
            for pid, p in self._persons.items()
        }
        self._path.write_text(json.dumps(data, indent=2))
