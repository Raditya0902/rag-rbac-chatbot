import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.limiter import limiter
from src.auth.jwt import create_access_token
from src.auth.users import authenticate_user

router = APIRouter()
log = structlog.get_logger()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


@router.post("/auth/login", response_model=LoginResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    user = authenticate_user(body.username, body.password)
    if not user:
        log.warning("login_failed", username=body.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user["username"], user["role"])
    log.info("login_success", username=user["username"], role=user["role"])
    return LoginResponse(access_token=token, username=user["username"], role=user["role"])
