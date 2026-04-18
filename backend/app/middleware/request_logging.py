import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.request_context import reset_request_id, set_request_id


access_logger = logging.getLogger("medgraphqa.access")
error_logger = logging.getLogger("medgraphqa.error")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, access_log_enabled: bool = True):
        super().__init__(app)
        self.access_log_enabled = access_log_enabled

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = set_request_id(request_id)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            error_logger.exception(
                "request failed method=%s path=%s client=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                request.client.host if request.client else "-",
                duration_ms,
            )
            reset_request_id(token)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id

        if self.access_log_enabled:
            access_logger.info(
                "method=%s path=%s status=%s client=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                request.client.host if request.client else "-",
                duration_ms,
            )

        reset_request_id(token)
        return response
