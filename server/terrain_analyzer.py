import numpy as np
import logging

logger = logging.getLogger(__name__)

# Heuristics for interpreting MiDaS depth gradients at 320x240
# MiDaS outputs disparity: higher values = closer, lower values = further.
# A sudden DROP in disparity means a sudden INCREASE in physical distance (a drop-off).
MIN_DROP_GRADIENT = 15.0  # Disparity drop indicating a significant step down
STAIRS_PATTERN_THRESHOLD = 3 # Number of alternating gradients that suggest stairs

def analyze_terrain(depth_map):
    """
    Analyzes the lower half of the depth map (where the ground is)
    to detect drop-offs, stairs, and uneven terrain.
    Returns: list of hazard dictionaries
    """
    hazards = []
    
    # We only care about the ground, which is the bottom 50% of the image.
    h, w = depth_map.shape
    ground_crop = depth_map[int(h * 0.5):, :]
    
    # Take a vertical gradient (difference between consecutive rows)
    # A positive gradient means the pixel below is closer (normal floor slope).
    # A strongly negative gradient means the pixel below is MUCH further away (a drop).
    vertical_diffs = np.diff(ground_crop, axis=0) # shape: (h/2 - 1, w)
    
    # We collapse the width by taking the median gradient per row to ignore noise/small objects
    row_gradients = np.median(vertical_diffs, axis=1) # shape: (h/2 - 1,)
    
    # 1. Detect Drop-offs (edges, curbs, potholes)
    # Look for rows where the gradient suddenly drops negatively.
    severe_drops = np.where(row_gradients < -MIN_DROP_GRADIENT)[0]
    
    if len(severe_drops) > 0:
        # The lowest row index in severe_drops is the highest up in the image (furthest away)
        # The higher the row index (closer to bottom of screen), the closer the drop is to the user.
        closest_drop_row = np.max(severe_drops)
        
        # Approximate distance to the drop-off.
        # Bottom of image (closest_drop_row == len) is ~0.1m.
        # Top of ground crop (closest_drop_row == 0) is ~3.0m.
        fraction_from_bottom = 1.0 - (closest_drop_row / len(row_gradients))
        drop_distance = round(max(0.1, fraction_from_bottom * 3.0), 2)
        
        hazards.append({
            "type": "step_down",
            "distance": drop_distance
        })
        logger.info(f"[TERRAIN] Detected step_down at ~{drop_distance}m")
        
        # If we found a drop, let's also check if there are multiple (stairs)
        # Look for a repeating pattern: drop, flat, drop, flat
        if len(severe_drops) >= STAIRS_PATTERN_THRESHOLD:
            # Check if they are somewhat evenly spaced (crude heuristic)
            hazards = [{
                "type": "stairs_down",
                "distance": drop_distance
            }]
            logger.info(f"[TERRAIN] Multiple drops detected -> stairs_down at ~{drop_distance}m")

    # 2. Could add uneven ground detection here based on variance in the row_gradients
    
    return hazards
