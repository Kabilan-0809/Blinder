from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import json
import asyncio
import time

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
            objects_info = process_frame(frame)
            
            # 2. Priority Danger Evaluation (<1m)
            danger_alert = check_immediate_danger(objects_info)
            if danger_alert:
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "text": danger_alert["instruction"]
                }))
                continue # Skip full scene LLM processing for this frame

            # 3. Scene Construction
            scene_json = build_scene(objects_info)
            
            # 4. Contextual Instruction Generation
            state = get_session(session_id)
            instruction_map = generate_guidance(scene_json, state)
            instruction_text = instruction_map["instruction"]
            
            # 5. Deduplication
            if instruction_text == state.get("last_instruction"):
                # Scene or LLM conclusion hasn't changed enough to warrant re-speaking
                continue
                
            # Update Redis-ready context 
            update_session(session_id, instruction_text, scene_json, objects_info)
            
            # Return final guidance
            await ws.send_text(json.dumps({
                "type": "scene",
                "text": instruction_text
            }))
            
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")