"""
ArthaMart — password hashing (PBKDF2-HMAC-SHA256).
No plaintext passwords ever touch the database.
"""
from __future__ import annotations

import hashlib
import hmac
import os

_ITERATIONS = 120_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'salt$hash' (hex). Random 16-byte salt per password."""
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return salt.hex() + "$" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify against 'salt$hash'."""
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return hmac.compare_digest(dk, expected)
