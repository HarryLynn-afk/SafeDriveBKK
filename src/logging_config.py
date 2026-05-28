"""
Centralised logging configuration for SafeDrive BKK.

Call ``setup_logging()`` once at application startup (inside ``create_app``).
After that, every module that does ``logging.getLogger(__name__)`` will
inherit the handlers and formatting configured here.

Environment variables
---------------------
LOG_LEVEL   : DEBUG | INFO | WARNING | ERROR | CRITICAL  (default: INFO)
LOG_FILE    : path to write rotating log file            (default: no file)
"""

import logging
import logging.handlers
from pathlib import Path

from .config import LOG_FILE, LOG_LEVEL


def setup_logging() -> None:
    """Configure root logger with console + optional rotating file handler.

    Safe to call multiple times — existing handlers are cleared first so
    calling ``create_app()`` in tests does not accumulate duplicate handlers.
    """
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Clear any handlers added by a previous call (e.g. during tests).
    root.handlers.clear()
    root.setLevel(level)

    # ── Console handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── Rotating file handler (opt-in via LOG_FILE env var) ──────────────────
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=5,               # keep last 5 rotated files
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Silence noisy third-party loggers at WARNING unless we're in DEBUG mode.
    if level > logging.DEBUG:
        for noisy in ("urllib3", "werkzeug", "googlemaps"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
