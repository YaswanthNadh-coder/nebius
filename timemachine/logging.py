"""Centralized logging configuration for timemachine."""

import logging
import sys

_LOGGER_NAME = "timemachine"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger for timemachine components."""
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure stream logging handler with standard formatting."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
