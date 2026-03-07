"""Persistent face database backed by the shared SQLite robot database.

Only embeddings (128-d float vectors) are stored — no raw images.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memory import DB_PATH, MemoryStore

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
        self._store = MemoryStore(path)

    # ------------------------------------------------------------------
    # Public API

    def enrol(self, name: str, embeddings: list[np.ndarray]) -> Person:
        """Add a new person with one or more face embeddings."""
        person_record = self._store.create_person(name)
        for embedding in embeddings:
            self._store.add_face_embedding(
                person_record.id,
                embedding,
                source="vision",
                threshold=SIMILARITY_THRESHOLD,
            )
        person = Person(
            id=person_record.id,
            name=person_record.display_name,
            enrolled_at=person_record.created_at,
            embeddings=embeddings,
        )
        print(f"[FaceDB] Enrolled '{name}' (id={person.id}, {len(embeddings)} embedding(s))")
        return person

    def identify(self, embedding: np.ndarray) -> Person | None:
        """Return the closest known person, or None if no match within threshold."""
        best_person: Person | None = None
        best_dist = float("inf")

        for record in self._store.list_face_embeddings():
            dist = float(np.linalg.norm(embedding - record.embedding))
            if dist < best_dist:
                best_dist = dist
                best_person = Person(
                    id=record.person_id,
                    name=record.display_name,
                    enrolled_at=record.enrolled_at,
                    embeddings=[record.embedding],
                )

        if best_dist <= SIMILARITY_THRESHOLD:
            if best_person is not None:
                self._store.touch_person(best_person.id)
            return best_person
        return None

    def list_persons(self) -> list[Person]:
        persons: list[Person] = []
        for record in self._store.list_persons():
            persons.append(
                Person(
                    id=record.id,
                    name=record.display_name,
                    enrolled_at=record.created_at,
                    embeddings=self._store.get_person_embeddings(record.id),
                )
            )
        return persons

    def remove(self, person_id: str) -> bool:
        person = self._store.get_person(person_id)
        if person and self._store.remove_person(person_id):
            name = person.display_name
            print(f"[FaceDB] Removed '{name}' (id={person_id})")
            return True
        return False
