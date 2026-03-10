from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import json
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("blind_ai_pipeline")

import sys
import os
sys.path.append(os.path.dirname(__file__))

from memory import get_session, update_session
from vision import process_frame
from scene_builder import build_scene
from danger_detector import check_immediate_danger
from instruction_engine import generate_guidance

app = FastAPI()

app.mount("/app", StaticFiles(directory="client", html=True), name="client")

@app.get("/")
async def get():
    return HTMLResponse(content="""
    <html>
        <head><title>Blind AI API</title></head>
        <body>
            <h1>Production Vision AI Server Running</h1>
            <p>Visit <a href="/app">/app</a></p>
        </body>
    </html>
    """)

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    session_id = "user_prod_1"
    
    # Track time to enforce max 2 fps
    last_processed_time = 0.0
    
    try:
        while True:
            data = await ws.receive_bytes()
            
            # Enforce latency optimization: 2fps (500ms between frames)
            current_time = time.time()
            if current_time - last_processed_time < 0.5:
                continue
                
            last_processed_time = current_time

            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            # 1. Vision Processing (Resize & object extraction)
            logger.info(f"Processing frame at {current_time}...")
            objects_info = process_frame(frame)
            logger.info(f"Vision extracted {len(objects_info)} objects.")
            
            # 2. Priority Danger Evaluation (<1m)
            danger_alert = check_immediate_danger(objects_info)
            if danger_alert:
                logger.warning(f"HIGH PRIORITY ALERT TRIGGERED: {danger_alert['instruction']}")
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "text": danger_alert["instruction"],
                    "objects": objects_info # Send objects back for UI
                }))
                continue # Skip full scene LLM processing for this frame

            # 3. Scene Construction
            scene_json = build_scene(objects_info)
            
            # 4. Contextual Instruction Generation
            state = get_session(session_id)
            instruction_map = generate_guidance(scene_json, state)
            instruction_text = instruction_map["instruction"]
            logger.info(f"Generated guidance: '{instruction_text}'")
            
            # 5. Semantic Deduplication
            if instruction_text == "SKIP" or instruction_text == state.get("last_instruction"):
                # Semantic logic decided user already knows this, skip TTS
                logger.info("Semantic deduplication triggered. Instruction skipped.")
                await ws.send_text(json.dumps({
                    "type": "debug",
                    "objects": objects_info 
                }))
                
                # We still update state to keep memory flowing with the silent scene
                update_session(session_id, "Silently observed scene.", scene_json, objects_info)
                continue
                
            # Update Redis-ready context 
            update_session(session_id, instruction_text, scene_json, objects_info)
            logger.info("Session state updated.")
            
            # Return final guidance
            logger.info("Sending instruction payload to client.")
            await ws.send_text(json.dumps({
                "type": "scene",
                "text": instruction_text,
                "objects": objects_info
            }))
            
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected.")