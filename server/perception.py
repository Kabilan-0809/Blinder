import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Reverting to YOLOv8n to maintain strict < 300ms total pipeline latency
model = YOLO("yolov8n.pt")

# Standardizing to 320x240 for 2fps performance and robotics norm
PROCESS_W = 320
PROCESS_H = 240

TARGET_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "bench", "couch", "bed", "dining table", "toilet",
    "fire hydrant", "stop sign", "parking meter", "refrigerator",
    "pole", "stairs", "door", "wall", "trash can"
}

def process_frame(frame):
    """
    Stage 2: Perception Module
    Detects relevant objects and extracts type, bbox, vertical position.
    """
    resized_frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
    
    # Run YOLO inference
    results = model(resized_frame, conf=0.25, verbose=False)
    
    objects_info = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            
            if center_x < PROCESS_W / 3:
                position = "left"
            elif center_x > 2 * PROCESS_W / 3:
                position = "right"
            else:
                position = "center"
                
            objects_info.append({
                "type": label,
                "bbox": [x1, y1, x2, y2],
                "position": position
            })
            
    return objects_info, resized_frame
