from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_frame(frame):

    results = model(frame)
    
    height, width, _ = frame.shape
    objects_info = []

    for r in results:
        for box in r.boxes:

            label = model.names[int(box.cls)]
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            if center_x < width / 3:
                direction = "left"
            elif center_x > 2 * width / 3:
                direction = "right"
            else:
                direction = "center"
                
            is_danger = label in ["car", "truck", "bus", "bicycle"]
            
            objects_info.append({
                "label": label,
                "direction": direction,
                "area": area,
                "is_danger": is_danger
            })

    return objects_info