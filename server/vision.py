import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# YOLOv8n — lightweight. Lower confidence to catch more objects in cluttered scenes.
model = YOLO("yolov8n.pt")

def process_frame(frame):
    """
    Detect objects in the frame. Lower confidence (0.15) to catch more objects
    in cluttered environments. Returns list of dicts with type, position, distance, bbox.
    """
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]
    total_area = height * width

    # Lower confidence threshold to 0.15 so cluttered messy rooms get detected properly
    results = model(frame, conf=0.15, verbose=False)

    objects_info = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            distance_proxy = round(area / total_area, 3)

            if center_x < width / 3:
                direction = "left"
            elif center_x > 2 * width / 3:
                direction = "right"
            else:
                direction = "center"

            objects_info.append({
                "type": label,
                "position": direction,
                "distance": distance_proxy,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

    labels = [f"{o['type']}({o['confidence']})" for o in objects_info]
    logger.info(f"YOLO detected: {labels}")
    return objects_info