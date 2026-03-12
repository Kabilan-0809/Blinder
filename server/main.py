import sys, os  # type: ignore
sys.path.append(os.path.dirname(__file__))

import logging  # type: ignore

# ── Rich terminal logging with clear visual markers per subsystem ─────────────
# Format: TIME  LEVEL  MODULE: emoji message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Export the FastAPI app
from websocket_server import app  # type: ignore
