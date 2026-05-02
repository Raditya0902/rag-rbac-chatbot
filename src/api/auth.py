import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from pydantic import BaseModel

from src.api.limiter import limiter
from src.auth.denylist import revoke_token
from src.auth.jwt import create_access_token, decode_token
from src.auth.users import authenticate_user

router = APIRouter()
_bearer = HTTPBearer()
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


@router.post("/auth/logout")
@limiter.limit("30/minute")
def logout(request: Request, credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    jti = payload.get("jti")
    exp = payload.get("exp")
    if jti and exp:
        revoke_token(jti, exp)
        log.info("logout", username=payload.get("sub"), jti=jti)

    return {"status": "logged out"}
