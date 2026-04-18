import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.security import hash_password, verify_password
from app.services.auth_repository import AuthRepository


logger = logging.getLogger("medgraphqa.auth")


@dataclass
class SessionUser:
    username: str
    is_admin: bool
    expire_at: datetime


class AuthService:
    def __init__(self, auth_repository: AuthRepository, ttl_minutes: int):
        self.auth_repository = auth_repository
        self.ttl = timedelta(minutes=ttl_minutes)

    def register(self, username: str, password: str) -> None:
        if self.auth_repository.user_exists(username):
            raise ValueError("用户名已存在")
        password_hash = hash_password(password)
        self.auth_repository.create_user(username=username, password_hash=password_hash)
        logger.info("user registered username=%s", username)

    def login(self, username: str, password: str) -> tuple[str, bool]:
        user = self.auth_repository.get_user(username)
        if not user or not user.get("is_active", True):
            logger.warning("login failed reason=user_not_found username=%s", username)
            raise ValueError("用户名或密码错误")
        if not verify_password(password, user["password_hash"]):
            logger.warning("login failed reason=bad_password username=%s", username)
            raise ValueError("用户名或密码错误")

        token = secrets.token_urlsafe(32)
        expire_at = datetime.now(timezone.utc) + self.ttl
        self.auth_repository.create_session(
            username=username, token=token, expire_at=expire_at
        )
        logger.info(
            "login success username=%s is_admin=%s",
            username,
            bool(user.get("is_admin", False)),
        )
        return token, bool(user.get("is_admin", False))

    def get_user_by_token(self, token: str) -> SessionUser | None:
        data = self.auth_repository.get_user_by_token(token)
        if not data:
            return None
        return SessionUser(
            username=str(data["username"]),
            is_admin=bool(data.get("is_admin", False)),
            expire_at=data["expire_at"],
        )

    def logout(self, token: str) -> None:
        self.auth_repository.revoke_token(token)
