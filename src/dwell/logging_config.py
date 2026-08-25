from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dwell.config import DwellConfig


def configure_logging(config: DwellConfig) -> None:
    """Route Dwell lifecycle records to a bounded local log file."""

    config.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dwell")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target = config.log_file.resolve(strict=False)
    for handler in list(logger.handlers):
        if getattr(handler, "_dwell_managed", False):
            existing = Path(getattr(handler, "baseFilename", "")).resolve(strict=False)
            if existing == target:
                return
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        config.log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler._dwell_managed = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
