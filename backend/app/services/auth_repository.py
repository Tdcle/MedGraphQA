import hashlib
import threading
from datetime import datetime, timezone

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class AuthRepository:
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

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def user_exists(self, username: str) -> bool:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM app_user WHERE username = %s",
                    (username,),
                )
                return cur.fetchone() is not None

    def create_user(
        self,
        username: str,
        password_hash: str,
        is_admin: bool = False,
    ) -> None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO app_user (username, password_hash, is_admin)
                    VALUES (%s, %s, %s)
                    """,
                    (username, password_hash, is_admin),
                )

    def get_user(self, username: str) -> dict | None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT username, password_hash, is_admin, is_active
                    FROM app_user
                    WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def create_session(
        self,
        username: str,
        token: str,
        expire_at: datetime,
    ) -> None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth_session (token_hash, username, expire_at)
                    VALUES (%s, %s, %s)
                    """,
                    (self.token_hash(token), username, expire_at),
                )

    def get_user_by_token(self, token: str) -> dict | None:
        self._ensure_open()
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT u.username, u.is_admin, s.expire_at
                    FROM auth_session s
                    JOIN app_user u ON u.username = s.username
                    WHERE s.token_hash = %s
                      AND s.revoked_at IS NULL
                      AND s.expire_at > %s
                      AND u.is_active = TRUE
                    """,
                    (self.token_hash(token), now),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    """
                    UPDATE auth_session
                    SET last_seen_at = now()
                    WHERE token_hash = %s
                    """,
                    (self.token_hash(token),),
                )
        return dict(row)

    def revoke_token(self, token: str) -> None:
        self._ensure_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE auth_session
                    SET revoked_at = now()
                    WHERE token_hash = %s AND revoked_at IS NULL
                    """,
                    (self.token_hash(token),),
                )
