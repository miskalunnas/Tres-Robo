"""SQLite-backed persistent context storage for the robot."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "robot.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PersonRecord:
    id: str
    display_name: str
    created_at: str
    last_seen_at: str | None
    notes: str
    preferences: dict[str, Any]


@dataclass
class FaceEmbeddingRecord:
    person_id: str
    display_name: str
    embedding: np.ndarray
    enrolled_at: str
    source: str
    threshold: float | None


@dataclass
class MemoryFactRecord:
    person_id: str | None
    category: str
    key: str
    value: str
    confidence: float
    updated_at: str


class MemoryStore:
    """Small repository around a local SQLite database."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Persons / identity
    # ------------------------------------------------------------------

    def create_person(self, display_name: str, *, person_id: str | None = None) -> PersonRecord:
        person_id = person_id or str(uuid.uuid4())[:8]
        now = _utc_now()
        self._execute(
            """
            INSERT INTO persons (id, display_name, created_at, last_seen_at, notes, preferences_json)
            VALUES (?, ?, ?, ?, '', '{}')
            """,
            (person_id, display_name, now, now),
        )
        return self.get_person(person_id)  # type: ignore[return-value]

    def get_person(self, person_id: str) -> PersonRecord | None:
        row = self._fetchone(
            """
            SELECT id, display_name, created_at, last_seen_at, notes, preferences_json
            FROM persons
            WHERE id = ?
            """,
            (person_id,),
        )
        return self._row_to_person(row)

    def list_persons(self) -> list[PersonRecord]:
        rows = self._fetchall(
            """
            SELECT id, display_name, created_at, last_seen_at, notes, preferences_json
            FROM persons
            ORDER BY display_name COLLATE NOCASE
            """
        )
        return [person for row in rows if (person := self._row_to_person(row)) is not None]

    def touch_person(self, person_id: str) -> None:
        self._execute(
            "UPDATE persons SET last_seen_at = ? WHERE id = ?",
            (_utc_now(), person_id),
        )

    def remove_person(self, person_id: str) -> bool:
        return self._execute("DELETE FROM persons WHERE id = ?", (person_id,)).rowcount > 0

    # ------------------------------------------------------------------
    # Face embeddings
    # ------------------------------------------------------------------

    def add_face_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        source: str = "vision",
        threshold: float | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO face_embeddings (person_id, embedding_json, enrolled_at, source, threshold)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_id,
                json.dumps(np.asarray(embedding, dtype=np.float32).tolist()),
                _utc_now(),
                source,
                threshold,
            ),
        )

    def list_face_embeddings(self) -> list[FaceEmbeddingRecord]:
        rows = self._fetchall(
            """
            SELECT fe.person_id, p.display_name, fe.embedding_json, fe.enrolled_at, fe.source, fe.threshold
            FROM face_embeddings fe
            JOIN persons p ON p.id = fe.person_id
            ORDER BY p.display_name COLLATE NOCASE, fe.id
            """
        )

        return [
            FaceEmbeddingRecord(
                person_id=row["person_id"],
                display_name=row["display_name"],
                embedding=np.array(json.loads(row["embedding_json"]), dtype=np.float32),
                enrolled_at=row["enrolled_at"],
                source=row["source"],
                threshold=row["threshold"],
            )
            for row in rows
        ]

    def get_person_embeddings(self, person_id: str) -> list[np.ndarray]:
        rows = self._fetchall(
            """
            SELECT embedding_json
            FROM face_embeddings
            WHERE person_id = ?
            ORDER BY id
            """,
            (person_id,),
        )
        return [np.array(json.loads(row["embedding_json"]), dtype=np.float32) for row in rows]

    # ------------------------------------------------------------------
    # Sessions / messages
    # ------------------------------------------------------------------

    def start_session(self, *, person_id: str | None, wake_word: str) -> str:
        session_id = str(uuid.uuid4())
        self._execute(
            """
            INSERT INTO sessions (id, person_id, started_at, wake_word)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, person_id, _utc_now(), wake_word),
        )
        if person_id:
            self.touch_person(person_id)
        return session_id

    def end_session(self, session_id: str, *, end_reason: str) -> None:
        self._execute(
            """
            UPDATE sessions
            SET ended_at = ?, end_reason = ?
            WHERE id = ?
            """,
            (_utc_now(), end_reason, session_id),
        )

    def update_session_summary(self, session_id: str, summary: str) -> None:
        """Store a one-sentence summary of the session (used for next-session context)."""
        self._execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary.strip()[:500], session_id),
        )

    def get_previous_session_summary(
        self, person_id: str | None, *, exclude_session_id: str
    ) -> str | None:
        """Return the summary of the most recent ended session for this person, if any."""
        if not person_id:
            return None
        row = self._fetchone(
            """
            SELECT summary FROM sessions
            WHERE person_id = ? AND ended_at IS NOT NULL
              AND id != ? AND summary IS NOT NULL AND summary != ''
            ORDER BY ended_at DESC
            LIMIT 1
            """,
            (person_id, exclude_session_id),
        )
        return row["summary"] if row else None

    def attach_person_to_session(self, session_id: str, person_id: str) -> None:
        self._execute(
            "UPDATE sessions SET person_id = ? WHERE id = ?",
            (person_id, session_id),
        )
        self.touch_person(person_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._execute(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, _utc_now()),
        )

    def get_session_messages(self, session_id: str, *, limit: int = 20) -> list[dict[str, str]]:
        rows = self._fetchall(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (session_id, limit),
        )
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def render_session_transcript(self, session_id: str, *, limit: int = 40) -> str:
        messages = self.get_session_messages(session_id, limit=limit)
        lines: list[str] = []
        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def upsert_memory_fact(
        self,
        *,
        person_id: str | None,
        category: str,
        key: str,
        value: str,
        confidence: float,
    ) -> None:
        now = _utc_now()
        self._execute(
            """
            INSERT INTO memory_facts (person_id, category, key, value, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, category, key)
            DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (person_id, category, key, value, confidence, now),
        )

    def get_memory_facts(self, person_id: str | None, *, limit: int = 8) -> list[MemoryFactRecord]:
        if person_id:
            person_rows = self._fetchall(
                """
                SELECT person_id, category, key, value, confidence, updated_at
                FROM memory_facts
                WHERE person_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (person_id, limit),
            )
            global_rows = self._fetchall(
                """
                SELECT person_id, category, key, value, confidence, updated_at
                FROM memory_facts
                WHERE person_id IS NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = sorted(
                [*person_rows, *global_rows],
                key=lambda row: row["updated_at"],
                reverse=True,
            )[:limit]
        else:
            rows = self._fetchall(
                """
                SELECT person_id, category, key, value, confidence, updated_at
                FROM memory_facts
                WHERE person_id IS NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        return [
            MemoryFactRecord(
                person_id=row["person_id"],
                category=row["category"],
                key=row["key"],
                value=row["value"],
                confidence=row["confidence"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def render_memory_context(self, person_id: str | None, *, limit: int = 8) -> str:
        facts = self.get_memory_facts(person_id, limit=limit)
        if not facts:
            return ""
        return "\n".join(
            f"- [{fact.category}] {fact.key}: {fact.value} (confidence {fact.confidence:.2f})"
            for fact in facts
        )

    # ------------------------------------------------------------------
    # Knowledge base (FTS)
    # ------------------------------------------------------------------

    def add_knowledge(self, source: str, content: str) -> None:
        """Add a text chunk to the searchable knowledge base."""
        content = content.strip()
        if not content:
            return
        self._execute(
            """
            INSERT INTO knowledge (source, content, created_at)
            VALUES (?, ?, ?)
            """,
            (source.strip()[:200], content[:10000], _utc_now()),
        )

    def search_knowledge(self, query: str, *, limit: int = 8) -> list[str]:
        """Full-text search over the knowledge base. Returns list of content strings.
        Table should be seeded via scripts/seed_knowledge.py for TRES/SFP content.
        """
        query = query.strip()
        if not query:
            return []
        # FTS5: escape double-quotes, truncate to avoid long-query issues
        fts_query = query.replace('"', '""').strip()[:500]
        if not fts_query:
            return []
        try:
            rows = self._fetchall(
                """
                SELECT k.content
                FROM knowledge k
                INNER JOIN (
                    SELECT rowid FROM knowledge_fts
                    WHERE knowledge_fts MATCH ?
                    LIMIT ?
                ) f ON k.id = f.rowid
                ORDER BY f.rowid
                """,
                (fts_query, limit),
            )
            return [row["content"] for row in rows]
        except sqlite3.OperationalError:
            # FTS syntax error or no match
            return []

    def clear_knowledge(self) -> int:
        """Remove all knowledge base entries. Returns number of rows deleted. For re-seeding."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM knowledge")
            self._conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Events / tool calls
    # ------------------------------------------------------------------

    def add_event(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        person_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO events (type, session_id, person_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                session_id,
                person_id,
                json.dumps(payload or {}, ensure_ascii=True),
                _utc_now(),
            ),
        )

    def log_tool_call(
        self,
        *,
        tool_name: str,
        input_payload: dict[str, Any] | None,
        output_summary: str,
        success: bool,
        session_id: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO tool_calls (session_id, tool_name, input_json, output_summary, success, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                tool_name,
                json.dumps(input_payload or {}, ensure_ascii=True),
                output_summary,
                int(success),
                duration_ms,
                _utc_now(),
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _configure_connection(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA temp_store = MEMORY")
            self._conn.execute("PRAGMA cache_size = -8000")
            self._conn.commit()

    def _execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(query, params)
            self._conn.commit()
            return cursor

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(query, params).fetchall()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    preferences_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'vision',
                    threshold REAL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    person_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    wake_word TEXT NOT NULL,
                    end_reason TEXT,
                    summary TEXT,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL,
                    UNIQUE(person_id, category, key),
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    session_id TEXT,
                    person_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    tool_name TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    output_summary TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    duration_ms INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    content,
                    source,
                    content='knowledge',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS knowledge_fts_ai AFTER INSERT ON knowledge BEGIN
                    INSERT INTO knowledge_fts(rowid, content, source) VALUES (new.id, new.content, new.source);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_fts_ad AFTER DELETE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, source) VALUES ('delete', old.id, old.content, old.source);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_fts_au AFTER UPDATE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, source) VALUES ('delete', old.id, old.content, old.source);
                    INSERT INTO knowledge_fts(rowid, content, source) VALUES (new.id, new.content, new.source);
                END;

                CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id
                    ON face_embeddings(person_id);

                CREATE INDEX IF NOT EXISTS idx_sessions_person_started
                    ON sessions(person_id, started_at DESC);

                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                    ON messages(session_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_memory_facts_person_updated
                    ON memory_facts(person_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_memory_facts_global_updated
                    ON memory_facts(updated_at DESC)
                    WHERE person_id IS NULL;

                CREATE INDEX IF NOT EXISTS idx_events_session_created
                    ON events(session_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_events_person_created
                    ON events(person_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_tool_calls_session_created
                    ON tool_calls(session_id, created_at DESC);
                """
            )
            self._conn.commit()
        self._migrate_sessions_summary()

    def _migrate_sessions_summary(self) -> None:
        """Add summary column to sessions if missing (for existing DBs)."""
        with self._lock:
            row = self._conn.execute(
                "PRAGMA table_info(sessions)"
            ).fetchall()
            columns = [r[1] for r in row]
            if "summary" not in columns:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
                self._conn.commit()

    @staticmethod
    def _row_to_person(row: sqlite3.Row | None) -> PersonRecord | None:
        if row is None:
            return None
        return PersonRecord(
            id=row["id"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            notes=row["notes"],
            preferences=json.loads(row["preferences_json"] or "{}"),
        )
