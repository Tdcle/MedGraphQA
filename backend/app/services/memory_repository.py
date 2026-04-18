import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


logger = logging.getLogger("medgraphqa.memory")


@dataclass
class UserMemory:
    id: int | None
    user_id: str
    memory_type: str
    memory_key: str
    value: dict
    text: str
    source: str
    confidence: float
    status: str = "active"
    expires_at: datetime | None = None
    metadata: dict | None = None

    def to_prompt_text(self) -> str:
        return self.text.strip()

    def to_log_dict(self) -> dict:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "value": self.value,
            "text": self.text,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata or {},
        }


class MemoryRepository:
    def __init__(
        self,
        dsn: str,
        min_size: int,
        max_size: int,
        timeout_seconds: int,
    ) -> None:
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout_seconds,
            open=False,
        )
        self._open_lock = threading.Lock()
        self._opened = False

    def _ensure_open(self) -> None:
        if self._opened:
            return
        with self._open_lock:
            if not self._opened:
                self._pool.open(wait=False)
                self._opened = True

    def close(self) -> None:
        if self._opened:
            self._pool.close()
            self._opened = False

    def list_active(
        self,
        user_id: str,
        memory_types: Sequence[str] | None = None,
        limit: int = 30,
    ) -> list[UserMemory]:
        params: list[object] = [user_id]
        type_filter = ""
        if memory_types:
            type_filter = "AND memory_type = ANY(%s)"
            params.append(list(memory_types))
        params.append(limit)

        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT *
                    FROM user_memory
                    WHERE user_id = %s
                      AND status = 'active'
                      AND (expires_at IS NULL OR expires_at > now())
                      {type_filter}
                    ORDER BY
                        CASE memory_type
                            WHEN 'allergy' THEN 1
                            WHEN 'pregnancy' THEN 2
                            WHEN 'chronic_disease' THEN 3
                            WHEN 'medication' THEN 4
                            WHEN 'profile' THEN 5
                            WHEN 'preference' THEN 6
                            ELSE 20
                        END,
                        last_seen_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_for_user(
        self,
        user_id: str,
        statuses: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[UserMemory]:
        params: list[object] = [user_id]
        status_filter = "AND status <> 'deleted'"
        if statuses:
            status_filter = "AND status = ANY(%s)"
            params.append(list(statuses))
        params.append(limit)

        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT *
                    FROM user_memory
                    WHERE user_id = %s
                      {status_filter}
                    ORDER BY
                        CASE status WHEN 'pending' THEN 1 WHEN 'active' THEN 2 ELSE 9 END,
                        last_seen_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def upsert(self, memory: UserMemory) -> UserMemory:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO user_memory (
                        user_id,
                        memory_type,
                        memory_key,
                        value,
                        text,
                        source,
                        confidence,
                        status,
                        expires_at,
                        metadata,
                        last_seen_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (user_id, memory_type, memory_key) DO UPDATE SET
                        value = EXCLUDED.value,
                        text = EXCLUDED.text,
                        source = EXCLUDED.source,
                        confidence = GREATEST(user_memory.confidence, EXCLUDED.confidence),
                        status = CASE
                            WHEN user_memory.status = 'active' THEN 'active'
                            ELSE EXCLUDED.status
                        END,
                        expires_at = EXCLUDED.expires_at,
                        metadata = user_memory.metadata || EXCLUDED.metadata,
                        last_seen_at = now()
                    RETURNING *
                    """,
                    (
                        memory.user_id,
                        memory.memory_type,
                        memory.memory_key,
                        Jsonb(memory.value),
                        memory.text,
                        memory.source,
                        memory.confidence,
                        memory.status,
                        memory.expires_at,
                        Jsonb(memory.metadata or {}),
                    ),
                )
                row = cur.fetchone()
        return self._row_to_memory(row)

    def set_status(self, user_id: str, memory_id: int, status: str) -> UserMemory | None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    UPDATE user_memory
                    SET status = %s, last_seen_at = now()
                    WHERE id = %s AND user_id = %s
                    RETURNING *
                    """,
                    (status, memory_id, user_id),
                )
                row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    def update(
        self,
        user_id: str,
        memory_id: int,
        text: str,
        value: dict,
    ) -> UserMemory | None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    UPDATE user_memory
                    SET text = %s,
                        value = %s,
                        source = 'user',
                        confidence = 1.0,
                        last_seen_at = now()
                    WHERE id = %s AND user_id = %s AND status <> 'deleted'
                    RETURNING *
                    """,
                    (text, Jsonb(value), memory_id, user_id),
                )
                row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    @staticmethod
    def _row_to_memory(row: dict) -> UserMemory:
        return UserMemory(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            memory_type=str(row["memory_type"]),
            memory_key=str(row["memory_key"]),
            value=dict(row.get("value") or {}),
            text=str(row["text"]),
            source=str(row["source"]),
            confidence=float(row["confidence"]),
            status=str(row["status"]),
            expires_at=row.get("expires_at"),
            metadata=dict(row.get("metadata") or {}),
        )
