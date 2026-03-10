import torch
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Use MiDaS small for fast depth estimation within the 300ms budget
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Loading MiDaS_small depth estimator on {device}...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def estimate_depth(frame, objects_info):
    """
    Stage 3: Depth Estimation
    Takes the BGR frame, generates a depth map, averages depth over bounding boxes,
    and calculates an approximate physical distance in meters.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        
        # Resize the prediction to match the frame resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth_map = prediction.cpu().numpy()
    
    for obj in objects_info:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        
        # Guard bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            obj["distance"] = 5.0 # default metric
            continue
            
        region = depth_map[y1:y2, x1:x2]
        
        # We take the 90th percentile depth reading (closer end of the bounding box)
        if region.size > 0:
            avg_disp = np.percentile(region, 90)
            
            # Inverse heuristic for pseudo-meters
            # MiDaS provides disparity; higher values mean closer.
            if avg_disp > 0:
                # 400 is an empirical constant for MiDaS small at 320x240. 
                # This will give ~1m for very close objects, ~5-10m for typical far background.
                distance = max(0.3, min(10.0, 400.0 / avg_disp))
            else:
                distance = 10.0
        else:
            distance = 5.0
            
        obj["distance"] = round(float(distance), 2)
        
    return objects_info, depth_map
