"""Logging setup for Astra Core.

Writes to stderr and a rotating-free plain log file under ``logs/``.
Secrets are never logged: this module only configures handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure root logging once and return the astra logger."""
    global _CONFIGURED
    logger = logging.getLogger("astra")
    if _CONFIGURED:
        logger.setLevel(level.upper())
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")

    stderr = logging.StreamHandler()
    stderr.setFormatter(formatter)
    logger.addHandler(stderr)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level.upper())
    _CONFIGURED = True
    return logger
