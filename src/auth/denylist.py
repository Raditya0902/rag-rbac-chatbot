"""Redis-backed JWT denylist.

A revoked token's jti is stored in Redis with a TTL equal to the token's
remaining lifetime. Checks fall back to False (token allowed) if Redis is
unreachable so an outage doesn't lock everyone out.
"""
import time

import structlog

from src.config import settings

log = structlog.get_logger()

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def revoke_token(jti: str, exp: int) -> None:
    """Write jti to the denylist with TTL = remaining token lifetime."""
    ttl = max(int(exp - time.time()), 1)
    try:
        _get_redis().setex(f"denylist:{jti}", ttl, "1")
    except Exception as exc:
        log.error("denylist_revoke_failed", jti=jti, error=str(exc))


def is_revoked(jti: str) -> bool:
    """Return True if this jti has been revoked. Fails closed: returns False on Redis error."""
    try:
        return _get_redis().exists(f"denylist:{jti}") > 0
    except Exception as exc:
        log.warning("denylist_check_failed", jti=jti, error=str(exc))
        return False
