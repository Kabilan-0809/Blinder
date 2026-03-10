from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("blind_ai_pipeline")

import sys
import os
sys.path.append(os.path.dirname(__file__))

from memory import get_session, update_session, should_skip
from vision import process_frame
from scene_builder import build_scene
from danger_detector import check_immediate_danger
from instruction_engine import generate_guidance

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html><body>
        <h1>Blind AI Server Running</h1>
        <p>Web App: <a href="/app">/app</a></p>
    </body></html>""")

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.client.host  # Unique per IP for multi-user support
    logger.info(f"Client connected: {session_id}")

    last_frame_time = 0.0

    try:
        while True:
            data = await ws.receive_bytes()

            # ── Rate Limit: process at max 2fps ──────────────────────────────
            now = time.time()
            if now - last_frame_time < 0.5:
                continue
            last_frame_time = now

            # ── Decode Frame ──────────────────────────────────────────────────
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Received invalid/empty frame, skipping.")
                continue

            # ── Stage 1: Vision ───────────────────────────────────────────────
            objects_info = process_frame(frame)
            logger.info(f"Detected {len(objects_info)} objects: {[o['type'] for o in objects_info]}")

            # ── Stage 2: Immediate Danger Check (bypasses LLM) ───────────────
            danger = check_immediate_danger(objects_info)
            if danger:
                logger.warning(f"DANGER ALERT → '{danger['instruction']}'")
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "text": danger["instruction"],
                    "objects": objects_info
                }))
                continue

            # ── Stage 3: Build Scene Representation ───────────────────────────
            scene_json = build_scene(objects_info)

            # ── Stage 4: Short-Circuit Empty Scenes WITHOUT LLM ───────────────
            if len(objects_info) == 0:
                state = get_session(session_id)
                # Only say "path is clear" if the last instruction was not the same
                last_instructions = [h["instruction"] for h in state["history"][-2:]]
                if "The path ahead is completely clear." not in last_instructions:
                    text = "The path ahead is completely clear."
                    logger.info(f"Empty scene → local response: '{text}'")
                    update_session(session_id, text, scene_json)
                    await ws.send_text(json.dumps({
                        "type": "scene",
                        "text": text,
                        "objects": []
                    }))
                else:
                    logger.info("Empty scene — already told user path is clear. Silently skipping.")
                    await ws.send_text(json.dumps({"type": "debug", "objects": []}))
                continue

            # ── Stage 5: Scene Hash Deduplication (skip unchanged scenes) ─────
            if should_skip(session_id, scene_json, min_cooldown_secs=5.0):
                logger.info("Scene unchanged within cooldown window. Skipping LLM call.")
                await ws.send_text(json.dumps({"type": "debug", "objects": objects_info}))
                continue

            # ── Stage 6: LLM Reasoning ────────────────────────────────────────
            state = get_session(session_id)
            result = generate_guidance(scene_json, state["history"])

            if result["skip"]:
                logger.info("LLM returned SKIP. Scene not meaningful enough to speak.")
                await ws.send_text(json.dumps({"type": "debug", "objects": objects_info}))
                continue

            instruction_text = result["instruction"]
            logger.info(f"→ SPEAKING: '{instruction_text}'")

            update_session(session_id, instruction_text, scene_json)

            await ws.send_text(json.dumps({
                "type": "scene",
                "text": instruction_text,
                "objects": objects_info
            }))

    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected.")