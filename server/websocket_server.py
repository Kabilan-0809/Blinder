"""
websocket_server.py

Thin WebSocket transport layer for the Blind AI navigation assistant.
All agent orchestration logic lives in agent/agent_controller.py.

WebSocket message protocol (client → server):
  { "type": "audio",   "data": "<base64 webm>" }
  { "type": "frame",   "data": "<base64 JPEG>" }
  { "type": "gps",     "lat": float, "lng": float }
  { "type": "setting", "danger_detection": bool }
  { "type": "profile", "name": "..." }
  { "type": "route",   "start": {...}, "dest": "..." }

WebSocket message protocol (server → client):
  { "type": "response",   "text": "...", "source": "...", "priority": int }
  { "type": "transcript", "text": "..." }
  { "type": "safety",     "text": "..." }
  { "type": "welcome",    "text": "...", "ask_name": bool }
  { "type": "ambient",    "text": "..." }
  { "type": "error",      "text": "..." }
"""

import base64  # type: ignore
import json  # type: ignore
import logging  # type: ignore

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore

from agent.agent_controller import AgentController  # type: ignore

logger = logging.getLogger("blind_ai")

app = FastAPI(title="Blind AI Navigation Assistant")
app.mount("/app", StaticFiles(directory="client", html=True), name="client")


@app.get("/")
async def root():  # type: ignore
    return HTMLResponse(  # type: ignore
        "<h1>Blind AI Navigation Assistant</h1>"
        "<p><a href='/app'>Open App</a></p>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WEBSOCKET HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/stream")
async def stream(ws: WebSocket):  # type: ignore
    await ws.accept()

    session_id: str = ws.headers.get("x-session-id") or (
        ws.client.host if ws.client else "unknown"
    )  # type: ignore
    logger.info(f"🔗 [CONNECT] session={session_id}")

    # Create agent controller for this session
    controller = AgentController(session_id)

    # Send welcome
    welcome = controller.get_welcome()
    await _send(ws, welcome)

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            raw = message.get("text") or ""
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except Exception:
                logger.warning("[WS] Bad JSON payload — skipping")
                continue

            msg_type = payload.get("type")

            # ── Settings ──────────────────────────────────────
            if msg_type == "setting":
                if "danger_detection" in payload:
                    controller.set_danger_detection(bool(payload["danger_detection"]))
                continue

            # ── Profile ───────────────────────────────────────
            if msg_type == "profile":
                controller.set_user_name(payload.get("name", ""))
                continue

            # ── Audio → Agent Pipeline ────────────────────────
            if msg_type == "audio":
                try:
                    result = await controller.process_audio(payload.get("data", ""))
                    for msg in result.messages:
                        await _send(ws, msg)
                except Exception as e:
                    logger.error(f"💥 [AUDIO ERROR] {e}", exc_info=True)
                    await _send(ws, {"type": "error", "text": "Sorry, something went wrong processing your voice."})

            # ── Frame → Agent Pipeline ────────────────────────
            elif msg_type == "frame":
                try:
                    frame_b64 = payload.get("data", "")
                    if frame_b64:
                        jpeg_bytes = base64.b64decode(frame_b64)
                        result = await controller.process_frame(jpeg_bytes)
                        for msg in result.messages:
                            await _send(ws, msg)
                except Exception as e:
                    logger.error(f"💥 [FRAME ERROR] {e}", exc_info=True)

            # ── GPS → Agent Pipeline ──────────────────────────
            elif msg_type == "gps":
                try:
                    lat = payload.get("lat")
                    lng = payload.get("lng")
                    if lat is not None and lng is not None:
                        result = await controller.process_gps(lat, lng)
                        for msg in result.messages:
                            await _send(ws, msg)
                except Exception as e:
                    logger.error(f"💥 [GPS ERROR] {e}", exc_info=True)

            # ── Route loading ─────────────────────────────────
            elif msg_type == "route":
                try:
                    start = payload.get("start")
                    dest = payload.get("dest") or payload.get("destination", "")
                    if not start or not dest:
                        await _send(ws, {"type": "error", "text": "Missing start or destination for route."})
                    else:
                        route_result = await controller.load_route(start, dest)
                        await _send(ws, route_result)
                except Exception as e:
                    logger.error(f"💥 [ROUTE ERROR] {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info(f"🔌 [DISCONNECT] session={session_id}")
    except Exception as e:
        logger.error(f"💥 [WS FATAL] {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _send(ws: WebSocket, data: dict):  # type: ignore
    try:
        await ws.send_text(json.dumps(data))
    except Exception as e:
        logger.warning(f"[WS] Send failed: {e}")
