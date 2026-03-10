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
logger = logging.getLogger("blind_ai")

import sys, os
sys.path.append(os.path.dirname(__file__))

from memory import get_session, update_session, scene_has_changed, is_within_cooldown
from vision import process_frame
from scene_builder import build_scene
from danger_detector import check_immediate_danger
from instruction_engine import generate_guidance

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse("<h1>Blind AI Running</h1><p><a href='/app'>Open App</a></p>")

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.client.host
    logger.info(f"[CONNECT] Client: {session_id}")

    last_frame_time = 0.0

    try:
        while True:
            data = await ws.receive_bytes()

            # Rate limit: 2fps max
            now = time.time()
            if now - last_frame_time < 0.5:
                continue
            last_frame_time = now

            # Decode JPEG
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # ── STAGE 1: Object Detection ─────────────────────────────────
            objects_info = process_frame(frame)
            logger.info(f"[VISION] {len(objects_info)} objects: {[o['type'] for o in objects_info]}")

            # ── STAGE 2: Immediate Danger (bypasses all logic) ────────────
            danger = check_immediate_danger(objects_info)
            if danger:
                logger.warning(f"[DANGER] → '{danger['instruction']}'")
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "text": danger["instruction"],
                    "objects": objects_info
                }))
                continue

            # ── STAGE 3: Build Scene ───────────────────────────────────────
            scene_json = build_scene(objects_info)
            state = get_session(session_id)
            is_first = state["is_first"]

            # ── STAGE 4: Empty Scene Short-Circuit (no LLM) ───────────────
            if len(objects_info) == 0:
                # Only speak if we haven't said "clear" in recent history
                last_msgs = [h["instruction"] for h in state["history"][-2:]]
                if "Path is clear, walk straight." not in last_msgs:
                    text = "Path is clear, walk straight."
                    logger.info(f"[CLEAR] → '{text}'")
                    update_session(session_id, text, objects_info)
                    await ws.send_text(json.dumps({"type": "scene", "text": text, "objects": []}))
                else:
                    await ws.send_text(json.dumps({"type": "debug", "objects": []}))
                continue

            # ── STAGE 5: Smart Deduplication ──────────────────────────────
            # Skip if: scene type-set is identical AND we're within cooldown
            # BUT always speak on the first message or if objects have changed
            scene_changed = scene_has_changed(session_id, objects_info)
            within_cooldown = is_within_cooldown(session_id)

            if not is_first and not scene_changed and within_cooldown:
                logger.info(f"[DEDUP] Same objects within cooldown. Skipping LLM.")
                await ws.send_text(json.dumps({"type": "debug", "objects": objects_info}))
                continue

            # ── STAGE 6: LLM Scene Description ────────────────────────────
            logger.info(f"[LLM] Calling Gemini (first={is_first}, changed={scene_changed})...")
            
            # Run the synchronous GenAPI call in a background thread so we don't block
            # the WebSocket loop from receiving new frames and running local danger checks!
            import asyncio
            result = await asyncio.to_thread(
                generate_guidance, 
                scene_json, 
                state["history"], 
                is_first=is_first
            )

            if result["skip"]:
                logger.info("[LLM] Returned SKIP.")
                # Still update object types so dedup works correctly
                state["last_object_types"] = set(o["type"] for o in objects_info)
                await ws.send_text(json.dumps({"type": "debug", "objects": objects_info}))
                continue

            instruction = result["instruction"]
            logger.info(f"[SPEAK] → '{instruction}'")
            update_session(session_id, instruction, objects_info)

            await ws.send_text(json.dumps({
                "type": "scene",
                "text": instruction,
                "objects": objects_info
            }))

    except WebSocketDisconnect:
        logger.info(f"[DISCONNECT] {session_id}")