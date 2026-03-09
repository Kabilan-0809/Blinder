def check_immediate_danger(objects_info):
    """
    If an obstacle distance < 1 meter (implied by distance area metric > 0.25 on a 320x240 frame),
    Returns an immediate STOP command that bypasses normal AI reasoning.
    """
    for obj in objects_info:
        if obj["distance"] > 0.25 and obj["type"] in ["car", "truck", "bus", "bicycle", "motorcycle", "person"]:
            return {
                "instruction": f"STOP. {obj['type'].capitalize()} crossing ahead."
            }
            
    return None
