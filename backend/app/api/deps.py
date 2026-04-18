from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.container import ServiceContainer
from app.services.auth_service import SessionUser


bearer_scheme = HTTPBearer(auto_error=False)


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    container: ServiceContainer = Depends(get_container),
) -> SessionUser:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录",
        )
    session = container.auth_service.get_user_by_token(credentials.credentials)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态已失效",
        )
    return session

