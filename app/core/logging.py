import logging
import sys

import structlog

from app.core.config import get_settings


def configure_logging() -> None:
    """Structured logs to stdout. Never log SQL text, DB errors, or secrets (Sec 9)."""
    settings = get_settings()
    logging.basicConfig(stream=sys.stdout, level=settings.log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
