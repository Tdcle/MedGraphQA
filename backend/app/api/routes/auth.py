from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import bearer_scheme, get_container, get_current_user
from app.core.container import ServiceContainer
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserProfile

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, container: ServiceContainer = Depends(get_container)):
    try:
        container.auth_service.register(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "注册成功"}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, container: ServiceContainer = Depends(get_container)):
    try:
        token, is_admin = container.auth_service.login(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return TokenResponse(access_token=token, username=payload.username, is_admin=is_admin)


@router.get("/me", response_model=UserProfile)
def me(current_user=Depends(get_current_user)):
    return UserProfile(username=current_user.username, is_admin=current_user.is_admin)


@router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    container: ServiceContainer = Depends(get_container),
):
    if credentials:
        container.auth_service.logout(credentials.credentials)
    return {"message": "退出登录成功"}
