"""
main.py — entry point for the Blind AI Navigation Assistant server.
Run with: uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import sys, os  # type: ignore
sys.path.insert(0, os.path.dirname(__file__))

import logging  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress noisy 3rd-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "multipart", "uvicorn.access"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Export the FastAPI app (uvicorn target)
from websocket_server import app  # type: ignore

__all__ = ["app"]
