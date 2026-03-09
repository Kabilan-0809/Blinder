import asyncio
import websockets
import cv2
import numpy as np

async def test_websocket():
    # create a dummy image frame (a black square)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    data = buffer.tobytes()

    uri = "ws://localhost:8000/stream"
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket.")
            await ws.send(data)
            print("Sent black frame...")
            
            response = await ws.recv()
            print(f"Received instruction: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
