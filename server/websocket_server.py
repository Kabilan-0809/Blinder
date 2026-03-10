from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import json
import time
import asyncio
import logging

from perception import process_frame
from depth_estimator import estimate_depth
from scene_builder import build_scene
from safety_engine import run_safety_engine
from navigation_engine import generate_navigation_instruction
from memory import update_session
from terrain_analyzer import analyze_terrain

logger = logging.getLogger("blind_ai_robotics")

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse("<h1>Blind Navigation System (Robotics Grade)</h1><p><a href='/app'>Open App</a></p>")

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.client.host
    logger.info(f"[CONNECT] Session: {session_id}")
    
    last_frame_time = 0.0
    
    try:
        while True:
            data = await ws.receive_bytes()
            
            # Enforce 2fps architecture (500ms intervals)
            now = time.time()
            if now - last_frame_time < 0.5:
                continue
            last_frame_time = now
            
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            start_t = time.time()
            
            # ── STAGE 1 & 2: Frame Ingestion & Perception Module ───────────
            objects_info, resized_frame = process_frame(frame)
            
            # ── STAGE 3: Depth Estimation (MiDaS) ──────────────────────────
            objects_info, depth_map = estimate_depth(resized_frame, objects_info)
            
            # ── STAGE 3.5: Terrain Hazard Detection ────────────────────────
            terrain_hazards = analyze_terrain(depth_map)
            
            # ── STAGE 4: Hardware-Level Safety Override ────────────────────
            safety_override = run_safety_engine(objects_info, terrain_hazards)
            
            if safety_override:
                instruction = safety_override
                logger.warning(f"[HARD SAFETY] → {instruction}")
            else:
                # ── STAGE 5: Semantic Reasoning Engine ──────────────────────
                scene_json = build_scene(objects_info, terrain_hazards)
                
                # Asynchronously call the reasoning engine so we don't block
                # the 2fps safety check loop
                result = await asyncio.to_thread(
                    generate_navigation_instruction, 
                    session_id, 
                    scene_json
                )
                
                instruction = result.get("instruction")
                
                if instruction:
                    # Update context for the next frame
                    update_session(session_id, instruction, scene_json, objects_info)
                    logger.info(f"[NAVIGATE] → {instruction}")
                else:
                    # Maintain silence if reasoning fails or skips
                    logger.info("[NAVIGATE] → (silence)")
            
            end_t = time.time()
            latency = (end_t - start_t) * 1000
            
            if latency > 300:
                logger.warning(f"Pipeline latency violated constraint: {latency:.1f}ms")
            else:
                logger.info(f"Pipeline latency: {latency:.1f}ms")
                
            
            if instruction:
                await ws.send_text(json.dumps({
                    "type": "scene",
                    "text": instruction,
                    "objects": objects_info
                }))

    except WebSocketDisconnect:
        logger.info(f"[DISCONNECT] Session: {session_id}")
