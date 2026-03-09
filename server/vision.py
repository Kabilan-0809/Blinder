import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_frame(frame):
    # Enforce 320x240 resolution to guarantee < 300ms latency on low end edge devices
    frame = cv2.resize(frame, (320, 240))
    
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
                "distance": float(round(distance_proxy, 3))
            })

    return objects_info