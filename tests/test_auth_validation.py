"""Login / signup validation & brute-force throttle tests."""
import pytest

from src.auth import store
from src.data import database


@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "auth.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    store.ensure_table(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# email / password / name validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "foo", "foo@", "@x.com", "a b@c.com",
                                 "a@b", "a@b.c", "a@@b.com", "a@b..com"])
def test_invalid_emails_rejected(conn, bad):
    assert store.validate_email(bad) is False
    with pytest.raises(ValueError):
        store.create_local_user(conn, bad, "X", "secret1")


@pytest.mark.parametrize("good", ["a@b.co", "you@example.com",
                                  "first.last+tag@sub.domain.org"])
def test_valid_emails_accepted(conn, good):
    assert store.validate_email(good) is True


def test_short_password_rejected(conn):
    with pytest.raises(ValueError, match="6"):
        store.create_local_user(conn, "p@x.com", "P", "12345")


def test_long_password_rejected(conn):
    with pytest.raises(ValueError):
        store.create_local_user(conn, "p2@x.com", "P", "x" * 129)


def test_empty_name_rejected(conn):
    with pytest.raises(ValueError, match="name"):
        store.create_local_user(conn, "n@x.com", "   ", "secret1")


def test_valid_signup_and_duplicate(conn):
    u = store.create_local_user(conn, "OK@X.com", "Ok User", "secret1")
    assert u["email"] == "ok@x.com"                 # normalised lowercase
    with pytest.raises(ValueError, match="exists"):
        store.create_local_user(conn, "ok@x.com", "Again", "secret1")


def test_login_wrong_password(conn):
    store.create_local_user(conn, "l@x.com", "L", "secret1")
    assert store.check_local_login(conn, "l@x.com", "wrong") is None
    assert store.check_local_login(conn, "l@x.com", "secret1") is not None


# ---------------------------------------------------------------------------
# brute-force throttle
# ---------------------------------------------------------------------------

def test_lockout_after_five_failures():
    em = "lock-me@x.com"
    store.clear_fails(em)
    assert not store.is_locked(em)
    for _ in range(store.MAX_FAILS):
        store.record_fail(em)
    assert store.is_locked(em)
    assert store.lock_remaining(em) > 0
    store.clear_fails(em)
    assert not store.is_locked(em)
