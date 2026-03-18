"""
logger.py

Centralized structured logging for the Vision Assistant.

Usage:
    from utils.logger import get_logger
    logger = get_logger("vision")
    logger.info("Object detected", extra={"object": "chair", "distance_m": 1.5})
"""

import sys  # type: ignore
import logging  # type: ignore
from datetime import datetime  # type: ignore

# ── Custom Formatter ──────────────────────────────────────────────────────────

class VisionAssistantFormatter(logging.Formatter):
    """
    Production-grade log formatter with emoji prefixes for quick scanning.
    """

    LEVEL_ICONS = {
        "DEBUG":    "🔍",
        "INFO":     "ℹ️ ",
        "WARNING":  "⚠️ ",
        "ERROR":    "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record) -> str:  # type: ignore
        icon = self.LEVEL_ICONS.get(record.levelname, "  ")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        module = record.name[:20].ljust(20)
        msg = record.getMessage()
        return f"{ts}  {icon} {record.levelname:<8s}  {module}  {msg}"


# ── Logger Factory ────────────────────────────────────────────────────────────

_initialized = False

def _init_logging(level: str = "INFO"):  # type: ignore
    """Initialize the root logger with our custom formatter."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(VisionAssistantFormatter())
    root.handlers = [handler]

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "multipart",
                  "uvicorn.access", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:  # type: ignore
    """
    Get a named logger for a subsystem.

    Args:
        name:  Subsystem name (e.g. "vision", "agent", "speech")
        level: Minimum log level for the root logger (first call only)

    Returns:
        Configured logging.Logger instance
    """
    _init_logging(level)
    return logging.getLogger(name)
