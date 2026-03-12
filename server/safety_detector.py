import cv2  # type: ignore
import numpy as np  # type: ignore
import logging  # type: ignore
from ultralytics import YOLO  # type: ignore
from depth_estimator import estimate_depth  # type: ignore

logger = logging.getLogger(__name__)

# YOLOv8n for speed - safety detection must never add latency to the vision loop
_yolo = YOLO("yolov8n.pt")
STOP_DISTANCE_M = 1.0

def run_safety_check(jpeg_bytes: bytes) -> str | None:  # type: ignore
    """
    Runs a lightweight obstacle check on a JPEG frame.
    Returns a STOP instruction string if something is within 1m, else None.
    Designed to run every frame in parallel with the vision/conversation loop.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None  # type: ignore
    
    frame = cv2.resize(frame, (320, 240))
    results = _yolo(frame, conf=0.25, verbose=False)
    
    objects = []
    for r in results:
        for box in r.boxes:  # type: ignore
            label = _yolo.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            pos = "left" if cx < 106 else ("right" if cx > 213 else "center")
            objects.append({"type": label, "bbox": [x1, y1, x2, y2], "position": pos})
    
    if not objects:
        return None  # type: ignore
    
    # Estimate depth for this frame
    try:
        objects, _ = estimate_depth(frame, objects)
    except Exception as e:
        logger.warning(f"[SAFETY] Depth estimation failed: {e}")
        return None  # type: ignore
    
    for obj in objects:
        dist = obj.get("distance", 10.0)
        if dist < STOP_DISTANCE_M:
            label = obj["type"].capitalize()
            logger.warning(f"[SAFETY] {label} at {dist:.1f}m → STOP")
            return f"Stop. {label} right ahead of you."  # type: ignore
    
    return None  # type: ignore
