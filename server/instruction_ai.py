def generate_instruction(objects_info, last_instruction):

    if not objects_info:
        instruction = "Path clear. Walk forward."
        if instruction == last_instruction:
            return None
        return instruction

    # Sort to prioritize danger first, then by the largest area (closest object)
    objects_info.sort(key=lambda x: (x["is_danger"], x["area"]), reverse=True)
    
    primary = objects_info[0]
    label = primary["label"].capitalize()
    dir = primary["direction"]
    danger = primary["is_danger"]
    
    # Generate < 7 words instructions
    if danger:
        if dir == "center":
            instruction = f"{label} ahead. Stop."
        elif dir == "left":
            instruction = f"{label} on left. Move right."
        else:
            instruction = f"{label} on right. Move left."
    else:
        if dir == "center":
            instruction = f"{label} ahead. Move aside."
        elif dir == "left":
            instruction = f"{label} on left. Move right."
        else:
            instruction = f"{label} on right. Move left."

    if instruction == last_instruction:
        return None

    return instruction