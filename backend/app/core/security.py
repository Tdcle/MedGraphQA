import hashlib
import hmac
import os
from typing import Tuple


def _derive(password: str, salt: bytes, rounds: int = 120_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = _derive(password, salt)
    return f"{salt.hex()}${digest.hex()}"


def _split_hash(encoded: str) -> Tuple[bytes, bytes]:
    salt_hex, digest_hex = encoded.split("$", 1)
    return bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt, expected = _split_hash(encoded)
    except ValueError:
        return False
    actual = _derive(password, salt)
    return hmac.compare_digest(actual, expected)

