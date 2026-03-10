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
from memory import get_session, update_session
from terrain_analyzer import analyze_terrain
from path_planner import analyze_free_space, determine_safe_direction
from route_navigation import load_route, get_next_navigation_step
from guidance_engine import fuse_guidance

logger = logging.getLogger("blind_ai_robotics")

app = FastAPI()
app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def root():
    return HTMLResponse("<h1>Blind Navigation System (Robotics Grade)</h1><p><a href='/app'>Open App</a></p>")

# ── HTTP REST endpoint: load a Google Maps route ────────────────────────────
@app.post("/route/load")
async def api_load_route(payload: dict):
    """
    Called once by the client to set a navigation destination.
    Payload: {"session_id": "...", "start": {"lat":x, "lng":y}, "destination": "Eiffel Tower, Paris"}
    """
    session_id = payload.get("session_id", "default")
    session_state = get_session(session_id)
    
    ok = load_route(
        start_location=payload["start"],
        destination=payload["destination"],
        session_state=session_state
    )
    if ok:
        return {"status": "ok", "steps": len(session_state["route_steps"])}
    return {"status": "error", "message": "Failed to load route. Check GOOGLE_MAPS_API_KEY."}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = ws.client.host
    logger.info(f"[CONNECT] Session: {session_id}")
    
    last_frame_time = 0.0
    
    try:
        while True:
            # ── DUAL MESSAGE HANDLING ──────────────────────────────────────
            # The WebSocket receives two types of messages:
            #   1. Binary JPEG bytes → camera frame for vision processing
            #   2. JSON Text string  → {"lat": x, "lng": y} GPS update
            #
            # We use receive() to detect the type dynamically.
            message = await ws.receive()
            state = get_session(session_id)
            
            # ── GPS LOCATION UPDATE (text frame) ────────────────────────
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    
                    # Route load request coming from client
                    if "destination" in payload:
                        ok = load_route(
                            start_location=payload["start"],
                            destination=payload["destination"],
                            session_state=state
                        )
                        await ws.send_text(json.dumps({
                            "type": "route_loaded",
                            "ok": ok,
                            "steps": len(state.get("route_steps", []))
                        }))
                        continue
                    
                    # Live GPS location update
                    if "lat" in payload and "lng" in payload:
                        user_location = {"lat": payload["lat"], "lng": payload["lng"]}
                        state["last_location"] = user_location
                        
                        nav_instruction = get_next_navigation_step(user_location, state)
                        
                        if nav_instruction:
                            logger.info(f"[GPS NAV] → {nav_instruction}")
                            await ws.send_text(json.dumps({
                                "type": "navigation",
                                "text": nav_instruction,
                                "distance_to_turn": state.get("distance_to_next_turn"),
                                "step": state.get("current_route_step", 0)
                            }))
                except Exception as e:
                    logger.warning(f"[GPS] Could not parse text payload: {e}")
                    
                continue  # Don't run vision on GPS frames

            # ── CAMERA FRAME PROCESSING (binary frame) ────────────────────────
            if "bytes" not in message:
                continue

            # Enforce 2fps architecture (500ms intervals)
            now = time.time()
            if now - last_frame_time < 0.5:
                continue
            last_frame_time = now
            
            frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            start_t = time.time()
            
            # ── STAGE 1 & 2: Frame Ingestion & Perception Module ───────────
            objects_info, resized_frame = process_frame(frame)
            
            # ── STAGE 3: Depth Estimation (MiDaS) ──────────────────────────
            objects_info, depth_map = estimate_depth(resized_frame, objects_info)
            
            # ── STAGE 3.5: Terrain Hazard Detection ────────────────────────
            terrain_hazards = analyze_terrain(depth_map)
            
            # ── STAGE 4: Hardware-Level Safety Check ───────────────────────
            safety_alert = run_safety_engine(objects_info, terrain_hazards)
            
            # ── STAGE 4.5: Walkable Path Detection ──────────────────────────
            vision_instruction = None
            if not safety_alert:
                zone_clearance = analyze_free_space(objects_info, depth_map)
                path_recommendation = determine_safe_direction(zone_clearance)
                
                # ── STAGE 5: Semantic Reasoning Engine ──────────────────────
                scene_json = build_scene(objects_info, terrain_hazards, path_recommendation)
                
                # Asynchronously call the reasoning engine so we don't block
                # the 2fps safety check loop
                result = await asyncio.to_thread(
                    generate_navigation_instruction, 
                    session_id, 
                    scene_json
                )
                vision_instruction = result.get("instruction")
                
                if vision_instruction:
                    update_session(session_id, vision_instruction, scene_json, objects_info)
                    logger.info(f"[VISION] → {vision_instruction}")

            # ── STAGE 6: GPS Route Step (last known state from GPS updates) ─
            nav_instruction = None
            if state.get("last_location") and state.get("route_steps"):
                nav_instruction = get_next_navigation_step(state["last_location"], state)
                
            # ── STAGE 7: Guidance Fusion ─────────────────────────────────
            fused = fuse_guidance(safety_alert, vision_instruction, nav_instruction)
            
            end_t = time.time()
            latency = (end_t - start_t) * 1000
            
            if latency > 300:
                logger.warning(f"Pipeline latency violated: {latency:.1f}ms")
            else:
                logger.info(f"Pipeline latency: {latency:.1f}ms")
                
            # Always send the full fused result on every camera frame
            await ws.send_text(json.dumps({
                "type": "scene",
                "instruction": fused["instruction"],
                "navigation": fused["navigation"],
                "objects": objects_info,
                "terrain_hazards": terrain_hazards
            }))

    except WebSocketDisconnect:
        logger.info(f"[DISCONNECT] Session: {session_id}")
