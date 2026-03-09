from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_frame(frame):

    results = model(frame)
    
    height, width, _ = frame.shape
    total_area = height * width
    
    objects_info = []
    immediate_threat = False

    for r in results:
        for box in r.boxes:

            label = model.names[int(box.cls)]
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            area_fraction = area / total_area
            
            if center_x < width / 3:
                direction = "left"
            elif center_x > 2 * width / 3:
                direction = "right"
            else:
                direction = "center"
                
            is_danger = label in ["car", "truck", "bus", "bicycle", "motorcycle"]
            
            # FAST TRACK LOGIC: If it's a danger object and it takes up >15% of the screen, it's very close!
            if is_danger and area_fraction > 0.15:
                immediate_threat = True
            
            objects_info.append({
                "label": label,
                "direction": direction,
                "area": area,
                "is_danger": is_danger,
                "area_fraction": area_fraction
            })

    return objects_info, immediate_threat