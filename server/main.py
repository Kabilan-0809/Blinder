import sys, os
sys.path.append(os.path.dirname(__file__))

import logging

# Central logging configuration for the entire robotics pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Export the FastAPI app from the new modular autonomous router
from websocket_server import app