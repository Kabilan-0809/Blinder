import asyncio
import websockets
import cv2
import numpy as np
import json
import time

async def test_websocket():
    # create a dummy image frame (a black square)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()

    uri = "ws://127.0.0.1:8000/stream"
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket.")
            
            # Send frame 1
            await ws.send(data)
            print("Sent frame 1...")
            response = await ws.recv()
            data_resp = json.loads(response)
            print(f"Received (Frame 1) [{data_resp.get('type')}]: {data_resp.get('text')}")
            
            # Wait to bypass the 2 FPS (500ms) governor
            await asyncio.sleep(0.6)
            
            # Send frame 2 (identical to frame 1)
            await ws.send(data)
            print("Sent frame 2 (identical)...")
            
            # The server will deduplicate the instruction since the scene and LLM context 
            # hasn't changed, meaning we shouldn't get a response. We will wait 2s and timeout.
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                print(f"Received (Frame 2): {response}")
            except asyncio.TimeoutError:
                print("Received no response for Frame 2 (Deduplication Working!)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
