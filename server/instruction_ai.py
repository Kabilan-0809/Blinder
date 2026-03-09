def generate_instruction(objects_info, immediate_threat, last_instruction):

    # Fast Track: Immediate Danger!
    if immediate_threat:
        return {"type": "alert", "text": "STOP! Danger ahead!"}

    # Slow Track: Scene Description
    if not objects_info:
        instruction = "Path clear. Walk forward."
        if instruction == last_instruction:
            return None
        return {"type": "scene", "text": instruction}

    # Sort to prioritize danger first, then by the largest area (closest object)
    objects_info.sort(key=lambda x: (x["is_danger"], x["area"]), reverse=True)
    
    primary = objects_info[0]
    label = primary["label"].capitalize()
    dir = primary["direction"]
    danger = primary["is_danger"]
    
    if danger:
        if dir == "center":
            instruction = f"{label} ahead. Careful."
        elif dir == "left":
            instruction = f"{label} on left. Move right."
        else:
            instruction = f"{label} on right. Move left."
    else:
        # Give context about non-danger items
        if dir == "center":
            instruction = f"{label} ahead. Move around it."
        elif dir == "left":
            instruction = f"{label} on left."
        else:
            instruction = f"{label} on right."

    if instruction == last_instruction:
        return None

    return {"type": "scene", "text": instruction}