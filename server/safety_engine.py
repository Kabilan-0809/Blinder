import logging

logger = logging.getLogger(__name__)

STOP_DISTANCE_METERS = 1.0

def run_safety_engine(objects_info, terrain_hazards=None):
    """
    Stage 4b: Safety Engine
    Hardware-level override: if ANY object is less than 1.0 meter away,
    or any terrain drop is less than 0.2 meters away, 
    bypass the reasoning AI entirely and generate a hard STOP instruction.
    """
    if terrain_hazards is None:
        terrain_hazards = []
        
    # Check terrain hazards first (falling is more immediate danger than bumping)
    for hazard in terrain_hazards:
        if hazard.get("distance", 10.0) < 0.2:
            hazard_type = hazard.get("type", "drop-off").replace("_", " ")
            logger.warning(f"[SAFETY ENGINE] Terrain risk! {hazard_type} at {hazard.get('distance')}m")
            return f"STOP. {hazard_type.capitalize()} ahead."

    # Check objects
    for obj in objects_info:
        dist = obj.get("distance", 10.0)
        
        if dist < STOP_DISTANCE_METERS:
            obj_type = obj.get("type", "object")
            logger.warning(f"[SAFETY ENGINE] Collision risk! {obj_type} at {dist}m")
            
            # Must remain short, punchy, and under 10 words
            return f"STOP. {obj_type.capitalize()} in front of you."
            
    return None
