import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)
model = YOLO("yolov8n.pt")

def process_frame(frame):
    # Process at higher 640x480 resolution to vastly improve object detection accuracy
    frame = cv2.resize(frame, (640, 480))
    
    results = model(frame)
    height, width, _ = frame.shape
    total_area = height * width
    
    objects_info = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            distance_proxy = area / total_area # Larger area = closer
            
            # Bucket direction
            if center_x < width / 3:
                direction = "left"
            elif center_x > 2 * width / 3:
                direction = "right"
            else:
                direction = "center"
            
            objects_info.append({
                "type": label,
                "position": direction,
                "distance": float(round(distance_proxy, 3)),
                "bbox": [x1, y1, x2, y2]
            })

    return objects_info