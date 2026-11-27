"""
ArthaMart — user store (local + Google accounts).
Table is created lazily so pre-existing databases upgrade in place.
"""
from __future__ import annotations

import datetime as _dt
import re
import sqlite3
import time

from ..config import DB_PATH
from .security import hash_password, verify_password

EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+-]+@(?!.*\.\.)[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*"
    r"\.[A-Za-z]{2,7}$")
PASS_MIN, PASS_MAX = 6, 128

# simple in-memory brute-force throttle: 5 fails in 5 min → 60 s lock
MAX_FAILS, FAIL_WINDOW, LOCK_SECS = 5, 300, 60
_FAILS: dict[str, list[float]] = {}


def validate_email(email: str) -> bool:
    return bool(EMAIL_RE.match((email or "").strip()))


def is_locked(email: str) -> bool:
    now = time.time()
    recent = [t for t in _FAILS.get(email.lower(), []) if now - t < FAIL_WINDOW]
    _FAILS[email.lower()] = recent
    return len(recent) >= MAX_FAILS


def lock_remaining(email: str) -> int:
    now = time.time()
    recent = [t for t in _FAILS.get(email.lower(), []) if now - t < FAIL_WINDOW]
    if len(recent) < MAX_FAILS:
        return 0
    return max(1, int(LOCK_SECS - (now - recent[0])))


def record_fail(email: str) -> None:
    _FAILS.setdefault(email.lower(), []).append(time.time())


def clear_fails(email: str) -> None:
    _FAILS.pop(email.lower(), None)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def ensure_table(conn: sqlite3.Connection | None = None) -> None:
    close = conn is None
    c = conn or _conn()
    try:
        c.execute("""
        CREATE TABLE IF NOT EXISTS app_users (
            user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            name          TEXT NOT NULL,
            password_hash TEXT,                 -- NULL for OAuth accounts
            provider      TEXT NOT NULL DEFAULT 'local',
            created_at    TEXT DEFAULT (datetime('now'))
        )""")
        c.commit()
    finally:
        if close:
            c.close()


def get_user(conn, email: str):
    ensure_table(conn)   # guarantee the table exists before querying
    return conn.execute("SELECT * FROM app_users WHERE lower(email) = ?",
                        (email.lower(),)).fetchone()


def create_local_user(conn, email: str, name: str, password: str) -> dict:
    email = (email or "").strip().lower()
    name = (name or "").strip()
    password = password or ""
    if not validate_email(email):
        raise ValueError("Enter a valid email address (e.g. you@example.com).")
    if not (PASS_MIN <= len(password) <= PASS_MAX):
        raise ValueError(f"Password must be {PASS_MIN}–{PASS_MAX} characters.")
    if not name:
        raise ValueError("Enter your name.")
    if get_user(conn, email):
        raise ValueError("An account with this email already exists.")
    conn.execute(
        "INSERT INTO app_users (email, name, password_hash, provider) "
        "VALUES (?, ?, ?, 'local')",
        (email, name.strip() or email.split("@")[0],
         hash_password(password)))
    conn.commit()
    return {"email": email, "name": name.strip() or email.split("@")[0],
            "provider": "local"}


def check_local_login(conn, email: str, password: str):
    row = get_user(conn, email)
    if row is None or row["provider"] != "local" or not row["password_hash"]:
        return None
    if verify_password(password, row["password_hash"]):
        return {"email": row["email"], "name": row["name"],
                "provider": "local"}
    return None


def upsert_oauth_user(conn, email: str, name: str, provider: str) -> dict:
    """Google (or other OAuth) accounts — no password stored."""
    email = email.strip().lower()
    row = get_user(conn, email)
    if row is None:
        conn.execute(
            "INSERT INTO app_users (email, name, password_hash, provider) "
            "VALUES (?, ?, NULL, ?)", (email, name, provider))
    else:
        conn.execute("UPDATE app_users SET name = ?, provider = ? "
                     "WHERE email = ?", (name, provider, email))
    conn.commit()
    return {"email": email, "name": name, "provider": provider}


def user_count(conn) -> int:
    ensure_table(conn)
    return conn.execute("SELECT COUNT(*) FROM app_users").fetchone()[0]
