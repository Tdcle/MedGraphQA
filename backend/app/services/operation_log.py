import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.services.metrics import observe_operation


def _format_fields(fields: dict) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    **fields,
) -> Iterator[dict]:
    started = time.perf_counter()
    result: dict = {}
    logger.info("operation=%s status=start %s", operation, _format_fields(fields))
    try:
        yield result
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        observe_operation(
            operation,
            "error",
            duration_ms / 1000,
            fields=fields,
            result=result,
            error_type=type(exc).__name__,
        )
        logger.exception(
            "operation=%s status=error duration_ms=%.2f error_type=%s %s",
            operation,
            duration_ms,
            type(exc).__name__,
            _format_fields(fields),
        )
        raise
    else:
        duration_ms = (time.perf_counter() - started) * 1000
        observe_operation(
            operation,
            "ok",
            duration_ms / 1000,
            fields=fields,
            result=result,
        )
        logger.info(
            "operation=%s status=ok duration_ms=%.2f %s %s",
            operation,
            duration_ms,
            _format_fields(fields),
            _format_fields(result),
        )
