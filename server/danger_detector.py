import logging

logger = logging.getLogger(__name__)

# Objects that can physically block or harm through contact
DANGER_TYPES = {"car", "truck", "bus", "bicycle", "motorcycle"}
# People are only dangerous if extremely close (higher threshold)
PERSON_DANGER_THRESHOLD = 0.45
# Vehicles and objects danger threshold (moderate proximity)
OBJECT_DANGER_THRESHOLD = 0.30

def check_immediate_danger(objects_info: list) -> dict | None:
    """
    Triggers an immediate spoken STOP if a hazard is dangerously close.
    Threshold is based on bounding box area fraction (bigger = closer).
    """
    for obj in objects_info:
        d = obj["distance"]
        t = obj["type"]
        
        if t == "person" and d > PERSON_DANGER_THRESHOLD:
            logger.warning(f"DANGER: Person very close (distance={d:.2f})")
            return {"instruction": "Watch out — someone is right in front of you."}
        
        if t in DANGER_TYPES and d > OBJECT_DANGER_THRESHOLD:
            logger.warning(f"DANGER: {t} very close (distance={d:.2f})")
            return {"instruction": f"Stop! There's a {t} right ahead of you."}
    
    return None
