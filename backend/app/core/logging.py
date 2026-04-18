import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from app.core.request_context import get_request_id


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=(
            "%(asctime)s %(levelname)s [%(name)s] "
            "[request_id=%(request_id)s] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    handler.addFilter(RequestIdFilter())
    return handler


def _build_file_handler(path, level: int, retention_days: int) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(path),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    handler.addFilter(RequestIdFilter())
    return handler


def _build_plain_file_handler(path, level: int, retention_days: int) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(path),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    return handler


def configure_logging(settings) -> None:
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.raiseExceptions = settings.debug

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handlers = [_build_console_handler(level)]
    if settings.log_file_enabled:
        handlers.append(
            _build_file_handler(
                settings.log_dir / "app.log", level, settings.log_retention_days
            )
        )

    for handler in handlers:
        root_logger.addHandler(handler)

    access_logger = logging.getLogger("medgraphqa.access")
    access_logger.handlers.clear()
    access_logger.setLevel(level)
    if settings.log_file_enabled:
        access_handler = _build_file_handler(
            settings.log_dir / "access.log", level, settings.log_retention_days
        )
        access_logger.addHandler(access_handler)

    chat_trace_logger = logging.getLogger("medgraphqa.chat_trace")
    chat_trace_logger.handlers.clear()
    chat_trace_logger.setLevel(level)
    chat_trace_logger.propagate = False
    if settings.chat_trace_enabled:
        chat_trace_handler = _build_plain_file_handler(
            settings.log_dir / settings.chat_trace_file,
            level,
            settings.log_retention_days,
        )
        chat_trace_logger.addHandler(chat_trace_handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)

    for logger_name in ("elastic_transport.transport", "elastic_transport"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
