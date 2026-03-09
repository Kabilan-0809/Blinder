from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import cv2

import sys
import os
sys.path.append(os.path.dirname(__file__))
from memory import *
from vision import process_frame
from instruction_ai import generate_instruction

app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Blind AI API</title>
        </head>
        <body>
            <h1>Blind Navigation AI Server Running</h1>
            <p>Connect to the websocket at <b>ws://localhost:8000/stream</b> to start sending image frames.</p>
        </body>
    </html>
    """)

@app.websocket("/stream")
async def stream(ws: WebSocket):

    await ws.accept()

    session_id = "user1"
    create_session(session_id)

    try:
        while True:

            data = await ws.receive_bytes()

            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            objects_info = process_frame(frame)

            last_instruction = get_last_instruction(session_id)

            instruction = generate_instruction(objects_info, last_instruction)

            if instruction:
                update_instruction(session_id, instruction)
                await ws.send_text(instruction)
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")