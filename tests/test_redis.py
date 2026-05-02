"""Tests for JWT denylist and Redis session history.

Redis is mocked — no live Redis instance required at test time.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from src.auth import denylist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_redis(monkeypatch) -> MagicMock:
    """Patch _get_redis() to return a MagicMock."""
    mock_r = MagicMock()
    monkeypatch.setattr(denylist, "_get_redis", lambda: mock_r)
    return mock_r


# ---------------------------------------------------------------------------
# revoke_token
# ---------------------------------------------------------------------------

def test_revoke_token_sets_key_with_ttl(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    future_exp = int(time.time()) + 3600
    denylist.revoke_token("some-jti", future_exp)
    mock_r.setex.assert_called_once()
    key, ttl, value = mock_r.setex.call_args[0]
    assert key == "denylist:some-jti"
    assert ttl > 0
    assert value == "1"


def test_revoke_token_ttl_at_least_one_when_expired(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    past_exp = int(time.time()) - 10  # already expired
    denylist.revoke_token("expired-jti", past_exp)
    _, ttl, _ = mock_r.setex.call_args[0]
    assert ttl >= 1


def test_revoke_token_logs_and_does_not_raise_on_redis_error(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    mock_r.setex.side_effect = ConnectionError("Redis down")
    denylist.revoke_token("jti", int(time.time()) + 60)  # must not raise


# ---------------------------------------------------------------------------
# is_revoked
# ---------------------------------------------------------------------------

def test_is_revoked_returns_true_for_revoked_jti(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    mock_r.exists.return_value = 1
    assert denylist.is_revoked("revoked-jti") is True
    mock_r.exists.assert_called_once_with("denylist:revoked-jti")


def test_is_revoked_returns_false_for_unknown_jti(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    mock_r.exists.return_value = 0
    assert denylist.is_revoked("unknown-jti") is False


def test_is_revoked_returns_false_on_redis_error(monkeypatch):
    mock_r = _mock_redis(monkeypatch)
    mock_r.exists.side_effect = ConnectionError("Redis down")
    # Must fail open — not raise and not block the user
    assert denylist.is_revoked("any-jti") is False


# ---------------------------------------------------------------------------
# JWT contains jti
# ---------------------------------------------------------------------------

def test_create_access_token_includes_jti():
    from src.auth.jwt import create_access_token, decode_token
    token = create_access_token("alice", "Engineering")
    payload = decode_token(token)
    assert "jti" in payload
    assert len(payload["jti"]) > 0


def test_each_token_has_unique_jti():
    from src.auth.jwt import create_access_token, decode_token
    t1 = decode_token(create_access_token("alice", "Engineering"))
    t2 = decode_token(create_access_token("alice", "Engineering"))
    assert t1["jti"] != t2["jti"]
