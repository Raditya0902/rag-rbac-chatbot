import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from pydantic import BaseModel

from src.api.limiter import limiter
from src.auth.denylist import is_revoked
from src.auth.jwt import decode_token
from src.auth.rbac import get_allowed_namespaces
from src.monitoring.langsmith_setup import log_feedback
from src.rag.chain import run_query, stream_query

router = APIRouter()
_bearer = HTTPBearer()
log = structlog.get_logger()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    run_id: str
    score: int  # 1 = thumbs up, 0 = thumbs down
    comment: str = ""


def _get_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    return {"username": payload["sub"], "role": payload["role"]}


def _history_key(username: str, session_id: str) -> str:
    """Redis history key for a session, private to the user who owns it.

    The session id comes from the client, so on its own it can't be trusted:
    another user could send it and read this user's conversation. Prefixing the
    authenticated username keeps each user's history separate.
    """
    return f"{username}:{session_id}"


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
@limiter.limit("15/minute")
def chat(request: Request, body: ChatRequest, user: dict = Depends(_get_user)):
    namespaces = get_allowed_namespaces(user["role"])
    session_id = body.session_id or str(uuid.uuid4())
    result = run_query(body.message, namespaces, _history_key(user["username"], session_id))
    return {"username": user["username"], "role": user["role"], **result, "session_id": session_id}


@router.post("/chat/stream")
@limiter.limit("15/minute")
def chat_stream(request: Request, body: ChatRequest, user: dict = Depends(_get_user)):
    namespaces = get_allowed_namespaces(user["role"])
    history_key = _history_key(user["username"], body.session_id or str(uuid.uuid4()))

    def event_generator():
        for token in stream_query(body.message, namespaces, history_key):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/feedback")
@limiter.limit("30/minute")
def feedback(request: Request, body: FeedbackRequest, user: dict = Depends(_get_user)):
    log_feedback(body.run_id, float(body.score), body.comment)
    label = "up" if body.score == 1 else "down"
    log.info("feedback", username=user["username"], run_id=body.run_id, score=label)
    return {"status": "ok"}
