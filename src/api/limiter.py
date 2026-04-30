from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.auth.jwt import decode_token


def _user_or_ip(request: Request) -> str:
    """Rate-limit key: username from JWT when available, IP otherwise."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            return decode_token(auth[7:])["sub"]
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_user_or_ip)
