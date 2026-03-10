import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# YOLOv8s (small) — much better accuracy than nano, auto-downloads on first run
# Nano: 3.2M params, Small: 11.2M params — ~2x slower but significantly more accurate
model = YOLO("yolov8s.pt")

# Frame dimensions YOLO processes — 640px wide to maximise accuracy while keeping speed acceptable
PROCESS_W = 640
PROCESS_H = 480


def process_frame(frame):
    """
    Detect objects in the frame using YOLOv8s.
    Returns: list of dicts with type, position, distance, confidence, bbox, fill_ratio
    """
    frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
    total_area = PROCESS_W * PROCESS_H

    # Extremely low confidence to catch heavily occluded/dark objects in a messy room
    results = model(frame, conf=0.10, verbose=False)

    objects_info = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            fill_ratio = round(area / total_area, 3)  # 0.0 (far) to 1.0 (fills screen)

            if center_x < PROCESS_W / 3:
                direction = "left"
            elif center_x > 2 * PROCESS_W / 3:
                direction = "right"
            else:
                direction = "center"

            objects_info.append({
                "type": label,
                "position": direction,
                "distance": fill_ratio,   # larger = closer
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    labels = [f"{o['type']}@{o['position']}(dist={o['distance']})" for o in objects_info]
    logger.info(f"[YOLO] {labels}")
    return objects_info