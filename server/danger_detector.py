import logging

logger = logging.getLogger(__name__)

# Vehicles: anything that could physically harm when moving
MOVING_HAZARDS = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Fill-ratio thresholds — how much of the camera frame the object occupies
# 0.0 = tiny in distance, 1.0 = fills the entire screen (touching your nose)
VEHICLE_DANGER_FILL  = 0.25   # Vehicles are dangerous even at moderate distance
PERSON_DANGER_FILL   = 0.50   # A person must be quite close to trigger stop
GENERIC_DANGER_FILL  = 0.75   # Other objects (chairs, tables, doors) only if filling 75% of screen

# If NO labelled object detected but something occupies too much screen => treat as wall
WALL_FILL_THRESHOLD  = 0.80   # Any large unlabelled mass = obstacle / wall


def check_immediate_danger(objects_info: list) -> dict | None:
    """
    Checks for immediate physical danger and returns a STOP instruction if one is found.
    Also detects walls/doors via high fill_ratio on ANY detected object, regardless of label.
    """
    # 1. Check labelled objects with specific thresholds
    for obj in objects_info:
        d = obj["distance"]   # fill_ratio from vision.py
        t = obj["type"]

        if t in MOVING_HAZARDS and d > VEHICLE_DANGER_FILL:
            logger.warning(f"[DANGER] Moving hazard near: {t} fill={d:.2f}")
            return {"instruction": f"Watch out! There is a {t} very close to you."}

        if t == "person" and d > PERSON_DANGER_FILL:
            logger.warning(f"[DANGER] Person right in front: fill={d:.2f}")
            return {"instruction": "Stop — someone is right in front of you."}

        # Generic large obstacle: chair, table, door, refrigerator, wall, etc.
        if d > GENERIC_DANGER_FILL:
            logger.warning(f"[DANGER] Large obstacle close: {t} fill={d:.2f}")
            return {"instruction": f"Careful, there's something right ahead of you — slow down."}

    # 2. Wall / flat surface detection:
    # If a single bounding box fills more than 30% of screen even for unknown labels
    for obj in objects_info:
        if obj["distance"] > WALL_FILL_THRESHOLD and obj["position"] == "center":
            logger.warning(f"[DANGER] Large mass in center: {obj['type']} fill={obj['distance']:.2f}")
            return {"instruction": "There's something directly ahead — stop or turn."}

    return None
