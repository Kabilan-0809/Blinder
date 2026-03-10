import numpy as np
import logging

logger = logging.getLogger(__name__)

# Heuristic constant for MiDaS disparity to meters (same as depth_estimator)
MIDAS_DISPARITY_CONST = 400.0

def analyze_free_space(objects_info, depth_map):
    """
    Slices the depth map into 3 vertical zones (left, center, right).
    Calculates the general "free clearance" in meters for each zone.
    Caps a zone's free clearance if a discrete YOLO object is closer than the background depth.
    """
    h, w = depth_map.shape
    
    # We only care about the floor/lower-mid area for walking clearance.
    # We'll ignore the top 30% (sky/ceilings)
    walking_crop = depth_map[int(h * 0.3):, :]
    
    zone_width = w // 3
    left_zone_disparity = walking_crop[:, :zone_width]
    center_zone_disparity = walking_crop[:, zone_width:2*zone_width]
    right_zone_disparity = walking_crop[:, 2*zone_width:]
    
    # Calculate base clearance of the room geometry using the median disparity.
    # We use the 75th percentile disparity (which favors closer things) to be safe.
    def disp_to_meters(disp_matrix):
        if disp_matrix.size == 0:
            return 10.0
        disp_val = np.percentile(disp_matrix, 75)
        if disp_val > 0:
            return max(0.3, min(10.0, MIDAS_DISPARITY_CONST / disp_val))
        return 10.0

    zone_clearance = {
        "left": disp_to_meters(left_zone_disparity),
        "center": disp_to_meters(center_zone_disparity),
        "right": disp_to_meters(right_zone_disparity)
    }
    
    # Now brutally cap the clearance if a YOLO object is sitting in that zone
    for obj in objects_info:
        pos = obj.get("position")
        dist = obj.get("distance", 10.0)
        
        if pos in zone_clearance:
            # If the object is closer than the room's background wall, cap it.
            if dist < zone_clearance[pos]:
                zone_clearance[pos] = dist
                
    return zone_clearance

def determine_safe_direction(zone_clearance):
    """
    Analyzes the 3 zones and returns the safest direction mathematically.
    """
    # Find the maximum clearance
    best_zone = max(zone_clearance, key=zone_clearance.get)
    best_dist = zone_clearance[best_zone]
    
    # If even the best zone has less than 1.5 meters of clearance, the whole path is narrow/blocked.
    if best_dist < 1.5:
        return {
            "free_space": "none",
            "confidence": 1.0 # 100% sure it's blocked
        }
        
    # Calculate confidence based on how much better the best zone is compared to the worst.
    worst_zone = min(zone_clearance, key=zone_clearance.get)
    worst_dist = zone_clearance[worst_zone]
    
    # Simple confidence: if best is 5m and worst is 1m, diff is 4m. High confidence.
    # If best is 5m and worst is 4m, confidence is lower.
    diff = best_dist - worst_dist
    
    # Normalize confidence to [0, 1] range. A diff of 3.0m is considered 100% (1.0)
    confidence = min(1.0, max(0.1, diff / 3.0))
    
    logger.info(f"[PATH PLANNER] Safest: {best_zone.upper()} ({best_dist:.1f}m vs worst {worst_dist:.1f}m)")
    
    return {
        "free_space": best_zone,
        "confidence": round(confidence, 2)
    }
